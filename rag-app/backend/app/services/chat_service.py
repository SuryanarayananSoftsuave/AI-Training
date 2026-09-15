from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Literal
from uuid import uuid4

from langfuse import Langfuse

from app.core.config import Settings
from app.ingestion.embedder import embed_passages
from app.llm.base import LLMClient
from app.llm.prompts import get_live_versions
from app.models.schemas import ChatStreamFinal, Citation, JudgeVerdict, Judgment, LLMProvider, RetrievalDebug
from app.retrieval.mmr import mmr_select
from app.retrieval.pipeline import rerank_candidates, retrieve_candidates
from app.retrieval.qdrant_store import QdrantStore
from app.trace.redact import redact
from app.trace.schema import ChatTrace, TracedCandidate
from app.trace.store import TraceStore

logger = logging.getLogger(__name__)

# One "delta" event per streamed answer-text chunk, then exactly one "final"
# event carrying everything else (citations, judgment, retrieval debug).
AnswerEvent = tuple[Literal["delta", "final"], "str | ChatStreamFinal"]

_CITATION_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# gpt-oss (via Groq) ignores the prompt's plain "[1]" instruction and cites
# in OpenAI's internal "harmony"/file-citation style instead, e.g.
# "...15 days 【1†L1-L3】." -- full-width brackets, a dagger, then a
# hallucinated line-range that doesn't correspond to anything in our numbered
# context. The leading digit before the dagger IS the real source number
# though, so this is matched as a second, independent pattern rather than
# trying to force one model family to stop using its own trained citation
# habit (prompt engineering alone was not reliable enough to suppress it).
_FULLWIDTH_CITATION_RE = re.compile(r"【(\d+)[^】]*】")


def _extract_used_indices(text: str, num_sources: int) -> list[int]:
    """Recover which numbered context sources the generator actually cited,
    now that `stream_answer` returns plain streamed text instead of a
    structured `used_source_indices` field (dropped because JSON-schema
    output and token streaming are mutually exclusive on both providers). A
    marker outside the valid source range is silently ignored -- the same
    safety property the old structured field relied on implicitly (a
    hallucinated index just wouldn't match a real citation).

    Matches `[1]`, `[1, 3, 4]` (Gemini combines several markers in one
    ASCII bracket), and `【1†L1-L3】` (gpt-oss's own citation style) -- each
    found via real live usage where a narrower regex silently extracted
    ZERO citations from an otherwise well-grounded, correctly-cited answer.
    """
    seen: list[int] = []
    for match in _CITATION_MARKER_RE.finditer(text):
        for num_str in match.group(1).split(","):
            idx = int(num_str.strip())
            if 1 <= idx <= num_sources and idx not in seen:
                seen.append(idx)
    for match in _FULLWIDTH_CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= num_sources and idx not in seen:
            seen.append(idx)
    return seen


_OUT_OF_SCOPE_MESSAGE = (
    "This question doesn't appear to be related to the HR policy knowledge base "
    "I have access to. Please contact your HR business partner for help with this."
)


def _out_of_scope_response(
    reason: str,
    mode: str,
    candidates_count: int,
    reranked_count: int,
    variants: list[str],
    generator_provider: LLMProvider,
    judge_provider: LLMProvider,
    generator_temperature: float,
    judge_temperature: float,
    use_mmr: bool,
    answer: str = _OUT_OF_SCOPE_MESSAGE,
) -> ChatStreamFinal:
    return ChatStreamFinal(
        answer=answer,
        citations=[],
        judgment=Judgment(verdict=JudgeVerdict.OUT_OF_SCOPE, confidence=0, claims=[], notes=reason),
        retrieval_debug=RetrievalDebug(
            mode=mode, candidates_after_fusion=candidates_count, reranked_returned=reranked_count,
            query_variants=variants or None,
            generator_provider=generator_provider, judge_provider=judge_provider,
            generator_temperature=generator_temperature, judge_temperature=judge_temperature,
            use_mmr=use_mmr,
        ),
    )


async def _filter_variants_by_similarity(original: str, variants: list[str], settings: Settings) -> list[str]:
    """Enforce that every generated variant actually preserves the original
    question's meaning, rather than trusting the prompt instruction alone --
    embeds the original and each variant and drops any variant whose cosine
    similarity to the original falls below the configured threshold. A
    variant that drifted into a broader, narrower, or unrelated question
    would otherwise silently pollute the retrieval pool with irrelevant
    chunks.
    """
    vectors = await asyncio.to_thread(embed_passages, settings.embedding_model_name, [original, *variants])
    original_vec = vectors[0]

    kept = []
    for variant, vec in zip(variants, vectors[1:]):
        similarity = sum(a * b for a, b in zip(original_vec, vec))  # both normalized -> dot product == cosine
        if similarity >= settings.query_expansion_similarity_threshold:
            kept.append(variant)
        else:
            logger.warning(
                "dropped query variant, drifted from original meaning (similarity %.3f < threshold %.2f): %r",
                similarity, settings.query_expansion_similarity_threshold, variant,
            )
    return kept


async def answer_query(
    query: str,
    use_keyword_search: bool,
    use_query_expansion: bool,
    use_mmr: bool,
    top_k: int,
    doc_ids: list[str] | None,
    settings: Settings,
    store: QdrantStore,
    generator: LLMClient,
    judge: LLMClient,
    generator_provider: LLMProvider,
    judge_provider: LLMProvider,
    generator_temperature: float,
    judge_temperature: float,
    trace_store: TraceStore | None = None,
    langfuse: Langfuse | None = None,
) -> AsyncIterator[AnswerEvent]:
    """Optionally expand the question into a few alternate phrasings (multi-
    query retrieval), retrieve for the original question and every variant
    IN PARALLEL, merge + dedupe the candidate pool, then rerank the merged
    pool against the ORIGINAL question (never the variants -- reranking must
    measure relevance to what the user actually asked) -> generate (via
    `generator`, STREAMED token-by-token) -> judge (via `judge`) --
    independently selected providers, so a Groq-written answer can be
    fact-checked by Gemini or vice versa.

    Yields ("delta", chunk) for each piece of generated answer text, then
    exactly one ("final", ChatStreamFinal) event. The no-candidates and
    off-topic-gate-#1 short-circuits fire before any generation call, so
    they yield only a single "final" event with no "delta"s at all.

    If `trace_store` is given, a `ChatTrace` is recorded in a `finally`
    block wrapping the whole function -- this fires on every exit path,
    including a client disconnecting mid-stream (Starlette calls
    `aclose()` on this generator, raising `GeneratorExit` at the current
    `yield`), so a trace always gets written even for a request that never
    got a "final" event, with whatever partial answer text had streamed so
    far.
    """
    trace_id = str(uuid4())
    started = time.monotonic()
    mode = "hybrid" if use_keyword_search else "semantic"

    # Optional, additive observability layer (self-hosted Langfuse) --
    # entirely separate from the ChatTrace/TraceStore recording below,
    # which is the Week-5-required, dependency-free, always-on local trace.
    # `langfuse is None` (the default, LANGFUSE_ENABLED=false) makes every
    # `root_span`-guarded block below a no-op. Uses the manual
    # start_observation()/.update()/.end() API rather than
    # start_as_current_observation()'s `with` block, since this function is
    # an async generator with several early `yield`+`return` points that a
    # single context manager can't cleanly wrap -- root_span is ended
    # exactly once, in the `finally` block below, alongside the
    # already-existing trace write.
    root_span = langfuse.start_observation(name="chat.answer_query", as_type="span") if langfuse is not None else None
    if root_span is not None:
        root_span.update(
            input={"query": query, "doc_ids": doc_ids},
            metadata={
                "mode": mode,
                "use_query_expansion": use_query_expansion,
                "use_mmr": use_mmr,
                "top_k": top_k,
                "generator_provider": generator_provider,
                "judge_provider": judge_provider,
            },
        )
    logger.info(
        "chat query: %r (mode=%s, query_expansion=%s, use_mmr=%s, top_k=%d, doc_ids=%s, generator=%s@%.1f, judge=%s@%.1f)",
        query, mode, use_query_expansion, use_mmr, top_k, doc_ids or "any",
        generator_provider, generator_temperature, judge_provider, judge_temperature,
    )

    # Populated progressively as the function runs; read back in the
    # `finally` block below regardless of which branch/exception exits the
    # function, so the trace reflects however far the request actually got.
    variants: list[str] = []
    candidates: list = []
    accumulated: list[str] = []
    trace_ranked: list[TracedCandidate] = []
    trace_judgment: Judgment | None = None
    trace_error: str | None = None
    timings_ms: dict[str, float] = {}

    try:
        if use_query_expansion:
            step_start = time.monotonic()
            raw_variants = await generator.expand_query(query, settings.query_expansion_variant_count, generator_temperature)
            logger.info("generated %d raw variant(s) (%.2fs): %s", len(raw_variants), time.monotonic() - step_start, raw_variants)

            step_start = time.monotonic()
            variants = await _filter_variants_by_similarity(query, raw_variants, settings)
            logger.info(
                "same-meaning check: kept %d/%d variant(s) (%.2fs): %s",
                len(variants), len(raw_variants), time.monotonic() - step_start, variants,
            )
            if root_span is not None:
                expansion_span = root_span.start_observation(
                    name="expand_query", as_type="generation", model=generator.generator_model_name
                )
                expansion_span.update(
                    input={"question": query}, output={"raw_variants": raw_variants, "kept_variants": variants},
                )
                expansion_span.end()

        all_queries = [query, *variants]

        step_start = time.monotonic()
        candidates, total_raw = await retrieve_candidates(
            all_queries, use_keyword_search, settings.first_stage_limit, doc_ids, settings, store,
            with_vectors=use_mmr,
        )
        timings_ms["retrieve"] = round((time.monotonic() - step_start) * 1000, 1)

        logger.info(
            "retrieved across %d query variant(s) via %s search: %d raw -> %d unique after dedup (%.2fs)",
            len(all_queries), mode, total_raw, len(candidates), time.monotonic() - step_start,
        )
        if root_span is not None:
            retrieve_span = root_span.start_observation(name="retrieve", as_type="span")
            retrieve_span.update(
                input={"queries": all_queries, "mode": mode},
                metadata={"raw_candidates": total_raw, "deduped_candidates": len(candidates)},
            )
            retrieve_span.end()

        if not candidates:
            logger.info("no candidates found for query %r -- returning no_answer", query)
            if root_span is not None:
                root_span.update(output={"verdict": "no_answer"})
            final = ChatStreamFinal(
                answer="I couldn't find anything in the indexed documents relevant to that question.",
                citations=[],
                judgment=Judgment(verdict=JudgeVerdict.NO_ANSWER, confidence=0, claims=[], notes="No candidates retrieved."),
                retrieval_debug=RetrievalDebug(
                    mode=mode, candidates_after_fusion=0, reranked_returned=0,
                    query_variants=variants or None,
                    generator_provider=generator_provider, judge_provider=judge_provider,
                    generator_temperature=generator_temperature, judge_temperature=judge_temperature,
                    use_mmr=use_mmr,
                ),
            )
            accumulated.append(final.answer)
            trace_judgment = final.judgment
            yield ("final", final)
            return

        step_start = time.monotonic()
        pool_size = settings.mmr_pool_size if use_mmr else top_k
        reranked_pool = await rerank_candidates(settings.reranker_model_name, query, candidates, pool_size)  # always against the ORIGINAL question

        # The single best relevance score is always the first pick MMR would
        # make too (see mmr_select's docstring), so this is captured from the
        # sorted-by-score pool BEFORE any MMR reordering below -- the off-topic
        # gate's correctness shouldn't depend on how MMR happens to reorder the
        # rest of the pool.
        top_score = reranked_pool[0][1]

        if use_mmr and len(reranked_pool) > top_k:
            mmr_indices = mmr_select(
                relevance_scores=[score for _point, score in reranked_pool],
                vectors=[point.vector["dense"] for point, _score in reranked_pool],
                k=top_k,
                lambda_mult=settings.mmr_lambda,
            )
            ranked = [reranked_pool[i] for i in mmr_indices]
        else:
            ranked = reranked_pool
        timings_ms["rerank"] = round((time.monotonic() - step_start) * 1000, 1)

        trace_ranked = [
            TracedCandidate(
                chunk_id=str(point.id), filename=point.payload["filename"],
                page_number=point.payload.get("page_number"), rerank_score=round(score, 4), cited=False,
            )
            for point, score in ranked
        ]

        logger.info(
            "reranked %d -> top %d (%.2fs)%s: %s",
            len(candidates), len(ranked), time.monotonic() - step_start, " [MMR]" if use_mmr else "",
            [(point.payload["filename"], point.payload.get("page_number"), round(score, 3)) for point, score in ranked],
        )
        if root_span is not None:
            rerank_span = root_span.start_observation(name="rerank", as_type="span")
            rerank_span.update(
                input={"pool_size": pool_size, "candidates": len(candidates)},
                metadata={"top_score": round(top_score, 4), "returned": len(ranked), "mmr_applied": use_mmr},
            )
            rerank_span.end()

        # Vector search always returns its nearest neighbors, even for a
        # completely unrelated question -- it never returns "nothing." The
        # reranker score is the actual relevance signal: if even the best match
        # scores below this floor, nothing retrieved is genuinely relevant, so
        # treat it as out-of-scope for this knowledge base rather than letting
        # the generator answer from weak, unrelated context. Skips both the
        # generate and judge calls entirely.
        if top_score < settings.off_topic_score_threshold:
            logger.info(
                "top rerank score %.3f below off-topic threshold %.2f -- treating query %r as out of scope",
                top_score, settings.off_topic_score_threshold, query,
            )
            if root_span is not None:
                root_span.update(output={"verdict": "out_of_scope", "reason": "score_below_threshold", "top_score": round(top_score, 4)})
            final = _out_of_scope_response(
                f"Best rerank score {top_score:.3f} was below the off-topic threshold {settings.off_topic_score_threshold:.2f}.",
                mode, len(candidates), len(ranked), variants, generator_provider, judge_provider,
                generator_temperature, judge_temperature, use_mmr,
            )
            accumulated.append(final.answer)
            trace_judgment = final.judgment
            yield ("final", final)
            return

        numbered_context = "\n\n".join(
            f"[{i + 1}] (source: {point.payload['filename']}, page {point.payload.get('page_number')})\n{point.payload['text']}"
            for i, (point, _score) in enumerate(ranked)
        )
        logger.debug("context sent to %s (%d chars):\n%s", generator_provider, len(numbered_context), numbered_context)

        gen_span = (
            root_span.start_observation(name="generate", as_type="generation", model=generator.generator_model_name)
            if root_span is not None
            else None
        )
        if gen_span is not None:
            gen_span.update(
                input={"question": query, "context_chars": len(numbered_context)},
                metadata={"provider": generator_provider, "temperature": generator_temperature},
            )

        step_start = time.monotonic()
        async for chunk in generator.stream_answer(query, numbered_context, generator_temperature):
            accumulated.append(chunk)
            yield ("delta", chunk)
        answer_text = "".join(accumulated)
        used_indices = _extract_used_indices(answer_text, len(ranked))
        timings_ms["generate"] = round((time.monotonic() - step_start) * 1000, 1)
        logger.info(
            "generated answer (%.2fs, %d chars, cited markers=%s): %r",
            time.monotonic() - step_start, len(answer_text), used_indices, answer_text,
        )
        if gen_span is not None:
            # Token usage isn't captured here -- neither LLMClient.stream_answer
            # implementation currently surfaces provider usage metadata from a
            # streamed response (Gemini/Groq only expose it reliably on
            # non-streamed calls or the final chunk in a form not yet parsed
            # here) -- a documented gap, not an oversight.
            gen_span.update(output=answer_text)
            gen_span.end()

        # The score gate is a cheap pre-filter, but it can be fooled by a
        # fluent, plausible-sounding question that happens to score just above
        # threshold against unrelated content (e.g. "tell me about the movie
        # VIP" against an HR policy KB). The generator, having actually read the
        # retrieved chunks, is the more reliable signal: if it cited NONE of
        # them, it found nothing usable -- treat that as out-of-scope too.
        # Unlike the other two gates, the answer text has ALREADY been streamed
        # to the caller by this point (streaming means we can't retroactively
        # unsend it), so this reuses the real streamed text as the shown answer
        # rather than replacing it with the canned out-of-scope message -- it
        # just skips the (billable) judge call and forces the verdict to
        # OUT_OF_SCOPE, so the badge never misleadingly reads "Grounded" for a
        # decline that happens to be a technically-true claim about the context.
        if not used_indices:
            logger.info("generator cited zero sources for query %r -- treating as out of scope", query)
            if root_span is not None:
                root_span.update(output={"verdict": "out_of_scope", "reason": "zero_citations", "answer": answer_text})
            final = _out_of_scope_response(
                "The generator found no retrieved chunk it could cite an answer from.",
                mode, len(candidates), len(ranked), variants, generator_provider, judge_provider,
                generator_temperature, judge_temperature, use_mmr,
                answer=answer_text,
            )
            trace_judgment = final.judgment
            yield ("final", final)
            return

        for i, traced in enumerate(trace_ranked):
            traced.cited = (i + 1) in used_indices

        citations = [
            Citation(
                marker=i + 1,
                doc_id=point.payload["doc_id"],
                filename=point.payload["filename"],
                page_number=point.payload.get("page_number"),
                chunk_id=str(point.id),
                snippet=point.payload["text"][:280],
                rerank_score=round(score, 4),
            )
            for i, (point, score) in enumerate(ranked)
            if (i + 1) in used_indices
        ]

        judge_span = (
            root_span.start_observation(name="judge", as_type="generation", model=judge.judge_model_name)
            if root_span is not None
            else None
        )
        if judge_span is not None:
            judge_span.update(
                input={"question": query, "answer": answer_text},
                metadata={"provider": judge_provider, "temperature": judge_temperature},
            )

        step_start = time.monotonic()
        try:
            # A real, live-observed failure mode: the judge call is an
            # external API boundary (a separate provider request from
            # generation), and a transient provider outage here (e.g. a 503
            # "high demand") shouldn't crash a request whose answer already
            # generated successfully. Degrade to a JUDGE_UNAVAILABLE verdict
            # instead -- both so the user still gets their answer, and so the
            # trace honestly records that this specific call failed, rather
            # than silently showing an incomplete judgment indistinguishable
            # from any other case.
            judgment = await judge.judge_answer(query, answer_text, numbered_context, judge_temperature)
            timings_ms["judge"] = round((time.monotonic() - step_start) * 1000, 1)
            logger.info(
                "judged answer (%.2fs): verdict=%s confidence=%d (%d/%d claims supported)",
                time.monotonic() - step_start, judgment.verdict.value, judgment.confidence,
                sum(c.supported for c in judgment.claims), len(judgment.claims),
            )
        except Exception as exc:
            timings_ms["judge"] = round((time.monotonic() - step_start) * 1000, 1)
            trace_error = f"{type(exc).__name__}: {exc}"
            logger.exception("judge call failed (trace_id=%s) -- degrading to JUDGE_UNAVAILABLE", trace_id)
            judgment = Judgment(
                verdict=JudgeVerdict.JUDGE_UNAVAILABLE, confidence=0, claims=[],
                notes=f"The judge model call failed: {trace_error}",
            )
            if judge_span is not None:
                judge_span.update(metadata={"error": trace_error})
        trace_judgment = judgment
        if judge_span is not None:
            judge_span.update(output=judgment.model_dump(mode="json"))
            judge_span.end()

        logger.info("chat query complete (%.2fs total): %r", time.monotonic() - started, query)

        if root_span is not None:
            root_span.update(output={
                "verdict": judgment.verdict.value, "confidence": judgment.confidence, "citations": len(citations),
            })

        yield ("final", ChatStreamFinal(
            answer=answer_text,
            citations=citations,
            judgment=judgment,
            retrieval_debug=RetrievalDebug(
                mode=mode,
                candidates_after_fusion=len(candidates),
                reranked_returned=len(ranked),
                query_variants=variants or None,
                generator_provider=generator_provider,
                judge_provider=judge_provider,
                generator_temperature=generator_temperature,
                judge_temperature=judge_temperature,
                use_mmr=use_mmr,
            ),
        ))
    except Exception as exc:
        # Catch-all so ANY unexpected crash in this pipeline (not just the
        # judge-call failure already handled above) is visible in the trace
        # rather than silently leaving `error` unset -- still re-raised
        # unchanged, so existing crash behavior/logging elsewhere is
        # untouched; this only adds visibility, it doesn't swallow anything.
        if trace_error is None:
            trace_error = f"{type(exc).__name__}: {exc}"
        if root_span is not None:
            root_span.update(output={"verdict": "error"}, metadata={"error": trace_error})
        raise
    finally:
        timings_ms["total"] = round((time.monotonic() - started) * 1000, 1)
        if root_span is not None:
            # Ended exactly once here, regardless of which branch above set
            # its output (success, either off-topic gate, no-candidates, or
            # the except block) -- and also covers a mid-stream client
            # disconnect (GeneratorExit), same as the ChatTrace write below.
            root_span.end()
        if trace_store is not None:
            try:
                trace_store.append(ChatTrace(
                    trace_id=trace_id,
                    timestamp=datetime.now(timezone.utc),
                    question=redact(query),
                    mode=mode,
                    use_query_expansion=use_query_expansion,
                    query_variants=variants,
                    use_mmr=use_mmr,
                    doc_ids_filter=doc_ids,
                    top_k=top_k,
                    generator_provider=generator_provider,
                    generator_model=generator.generator_model_name,
                    generator_temperature=generator_temperature,
                    judge_provider=judge_provider,
                    judge_model=judge.judge_model_name,
                    judge_temperature=judge_temperature,
                    prompt_versions=get_live_versions(),
                    candidates_after_fusion=len(candidates),
                    ranked_candidates=trace_ranked,
                    answer=redact("".join(accumulated)),
                    judgment=trace_judgment,
                    error=trace_error,
                    timings_ms=timings_ms,
                ))
            except Exception:
                logger.exception("failed to record chat trace (trace_id=%s) -- continuing without it", trace_id)
