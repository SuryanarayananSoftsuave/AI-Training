from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import models

from app.core.config import Settings
from app.ingestion.embedder import embed_passages, embed_query
from app.ingestion.sparse import embed_sparse_query
from app.llm.base import LLMClient
from app.models.schemas import ChatResponse, Citation, JudgeVerdict, Judgment, LLMProvider, RetrievalDebug
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


def _retrieve_for_query(
    query_text: str,
    use_keyword_search: bool,
    limit: int,
    doc_ids: list[str] | None,
    settings: Settings,
    store: QdrantStore,
) -> list[models.ScoredPoint]:
    """Embed + search for a single query string -- dense-only, or dense+sparse
    fused with RRF when hybrid search is on. Runs once per query variant.
    """
    dense_vector = embed_query(settings.embedding_model_name, query_text)
    sparse_vector = embed_sparse_query(settings.sparse_model_name, query_text) if use_keyword_search else None
    return store.search(dense_vector=dense_vector, sparse_vector=sparse_vector, limit=limit, doc_ids=doc_ids)


_OUT_OF_SCOPE_MESSAGE = (
    "This question doesn't appear to be related to the customer support knowledge base "
    "I have access to. Please contact an administrator or your support team for help with this."
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
) -> ChatResponse:
    return ChatResponse(
        answer=_OUT_OF_SCOPE_MESSAGE,
        citations=[],
        judgment=Judgment(verdict=JudgeVerdict.OUT_OF_SCOPE, confidence=0, claims=[], notes=reason),
        retrieval_debug=RetrievalDebug(
            mode=mode, candidates_after_fusion=candidates_count, reranked_returned=reranked_count,
            query_variants=variants or None,
            generator_provider=generator_provider, judge_provider=judge_provider,
            generator_temperature=generator_temperature, judge_temperature=judge_temperature,
        ),
    )


def _filter_variants_by_similarity(original: str, variants: list[str], settings: Settings) -> list[str]:
    """Enforce that every generated variant actually preserves the original
    question's meaning, rather than trusting the prompt instruction alone --
    embeds the original and each variant and drops any variant whose cosine
    similarity to the original falls below the configured threshold. A
    variant that drifted into a broader, narrower, or unrelated question
    would otherwise silently pollute the retrieval pool with irrelevant
    chunks.
    """
    vectors = embed_passages(settings.embedding_model_name, [original, *variants])
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


def answer_query(
    query: str,
    use_keyword_search: bool,
    use_query_expansion: bool,
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
) -> ChatResponse:
    """Optionally expand the question into a few alternate phrasings (multi-
    query retrieval), retrieve for the original question and every variant
    IN PARALLEL, merge + dedupe the candidate pool, then rerank the merged
    pool against the ORIGINAL question (never the variants -- reranking must
    measure relevance to what the user actually asked) -> generate (via
    `generator`) -> judge (via `judge`) -- independently selected providers,
    so a Groq-written answer can be fact-checked by Gemini or vice versa.
    """
    started = time.monotonic()
    mode = "hybrid" if use_keyword_search else "semantic"
    logger.info(
        "chat query: %r (mode=%s, query_expansion=%s, top_k=%d, doc_ids=%s, generator=%s@%.1f, judge=%s@%.1f)",
        query, mode, use_query_expansion, top_k, doc_ids or "any",
        generator_provider, generator_temperature, judge_provider, judge_temperature,
    )

    variants: list[str] = []
    if use_query_expansion:
        step_start = time.monotonic()
        raw_variants = generator.expand_query(query, settings.query_expansion_variant_count, generator_temperature)
        logger.info("generated %d raw variant(s) (%.2fs): %s", len(raw_variants), time.monotonic() - step_start, raw_variants)

        step_start = time.monotonic()
        variants = _filter_variants_by_similarity(query, raw_variants, settings)
        logger.info(
            "same-meaning check: kept %d/%d variant(s) (%.2fs): %s",
            len(variants), len(raw_variants), time.monotonic() - step_start, variants,
        )

    all_queries = [query, *variants]

    step_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        futures = [
            executor.submit(
                _retrieve_for_query, q, use_keyword_search, settings.first_stage_limit, doc_ids, settings, store
            )
            for q in all_queries
        ]
        per_query_results = [future.result() for future in futures]

    seen_ids: set = set()
    candidates: list[models.ScoredPoint] = []
    total_raw = 0
    for result_list in per_query_results:
        total_raw += len(result_list)
        for point in result_list:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                candidates.append(point)

    logger.info(
        "retrieved across %d query variant(s) via %s search: %d raw -> %d unique after dedup (%.2fs)",
        len(all_queries), mode, total_raw, len(candidates), time.monotonic() - step_start,
    )

    if not candidates:
        logger.info("no candidates found for query %r -- returning no_answer", query)
        return ChatResponse(
            answer="I couldn't find anything in the indexed documents relevant to that question.",
            citations=[],
            judgment=Judgment(verdict=JudgeVerdict.NO_ANSWER, confidence=0, claims=[], notes="No candidates retrieved."),
            retrieval_debug=RetrievalDebug(
                mode=mode, candidates_after_fusion=0, reranked_returned=0,
                query_variants=variants or None,
                generator_provider=generator_provider, judge_provider=judge_provider,
                generator_temperature=generator_temperature, judge_temperature=judge_temperature,
            ),
        )

    step_start = time.monotonic()
    texts = [candidate.payload["text"] for candidate in candidates]
    scores = rerank(settings.reranker_model_name, query, texts)  # always against the ORIGINAL question
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    logger.info(
        "reranked %d -> top %d (%.2fs): %s",
        len(candidates), len(ranked), time.monotonic() - step_start,
        [(point.payload["filename"], point.payload.get("page_number"), round(score, 3)) for point, score in ranked],
    )

    # Vector search always returns its nearest neighbors, even for a
    # completely unrelated question -- it never returns "nothing." The
    # reranker score is the actual relevance signal: if even the best match
    # scores below this floor, nothing retrieved is genuinely relevant, so
    # treat it as out-of-scope for this knowledge base rather than letting
    # the generator answer from weak, unrelated context. Skips both the
    # generate and judge calls entirely.
    top_score = ranked[0][1]
    if top_score < settings.off_topic_score_threshold:
        logger.info(
            "top rerank score %.3f below off-topic threshold %.2f -- treating query %r as out of scope",
            top_score, settings.off_topic_score_threshold, query,
        )
        return _out_of_scope_response(
            f"Best rerank score {top_score:.3f} was below the off-topic threshold {settings.off_topic_score_threshold:.2f}.",
            mode, len(candidates), len(ranked), variants, generator_provider, judge_provider,
            generator_temperature, judge_temperature,
        )

    numbered_context = "\n\n".join(
        f"[{i + 1}] (source: {point.payload['filename']}, page {point.payload.get('page_number')})\n{point.payload['text']}"
        for i, (point, _score) in enumerate(ranked)
    )
    logger.debug("context sent to %s (%d chars):\n%s", generator_provider, len(numbered_context), numbered_context)

    step_start = time.monotonic()
    answer_text, used_indices = generator.generate_answer(query, numbered_context, generator_temperature)
    logger.info(
        "generated answer (%.2fs, %d chars, cited markers=%s): %r",
        time.monotonic() - step_start, len(answer_text), used_indices, answer_text,
    )

    # The score gate is a cheap pre-filter, but it can be fooled by a
    # fluent, plausible-sounding question that happens to score just above
    # threshold against unrelated content (e.g. "tell me about the movie
    # VIP" against a customer-support KB). The generator, having actually
    # read the retrieved chunks, is the more reliable signal: if it cited
    # NONE of them, it found nothing usable -- treat that as out-of-scope
    # too, rather than showing the model's own decline text as a "Grounded"
    # answer (technically true, but reads to the user like a real answer).
    if not used_indices:
        logger.info("generator cited zero sources for query %r -- treating as out of scope", query)
        return _out_of_scope_response(
            "The generator found no retrieved chunk it could cite an answer from.",
            mode, len(candidates), len(ranked), variants, generator_provider, judge_provider,
            generator_temperature, judge_temperature,
        )

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

    step_start = time.monotonic()
    judgment = judge.judge_answer(query, answer_text, numbered_context, judge_temperature)
    logger.info(
        "judged answer (%.2fs): verdict=%s confidence=%d (%d/%d claims supported)",
        time.monotonic() - step_start, judgment.verdict.value, judgment.confidence,
        sum(c.supported for c in judgment.claims), len(judgment.claims),
    )

    logger.info("chat query complete (%.2fs total): %r", time.monotonic() - started, query)

    return ChatResponse(
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
        ),
    )
