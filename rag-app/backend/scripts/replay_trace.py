"""Week 5 deliverable: replays one trace from the trace record ALONE (no
memory of the original request needed) and prints the original answer
alongside the replayed one -- proof that a trace is genuinely enough to
reconstruct a request. Explicitly names anything that could not be
reconstructed, rather than silently glossing over it.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/replay_trace.py <trace_id>
"""
from __future__ import annotations

import argparse
import asyncio

from app.core.config import get_settings
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.retrieval.qdrant_store import QdrantStore
from app.trace.store import TraceStore


def _build_generator(provider: str, model: str, settings) -> GeminiClient | GroqClient:
    if provider == "gemini":
        return GeminiClient(settings.gemini_api_key, model, model)
    if provider == "groq":
        return GroqClient(settings.groq_api_key, model, model)
    raise ValueError(f"unknown provider {provider!r}")


async def main(trace_id: str) -> None:
    settings = get_settings()
    trace_store = TraceStore(settings.trace_success_log_path, settings.trace_failure_log_path)
    trace = trace_store.get_by_id(trace_id)
    if trace is None:
        print(f"No trace found with id {trace_id!r} in {settings.trace_success_log_path} or {settings.trace_failure_log_path}.")
        return

    print(f"=== Trace {trace.trace_id} ({trace.timestamp.isoformat()}) ===")
    print(f"Question: {trace.question!r}")
    print(f"Generator: {trace.generator_provider}/{trace.generator_model} @ {trace.generator_temperature}")
    print(f"Prompt version (generation): {trace.prompt_versions.get('generation')}")
    print()

    missing_notes: list[str] = []

    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    chunk_ids = [c.chunk_id for c in trace.ranked_candidates]
    records = await store.get_by_ids(chunk_ids) if chunk_ids else []
    records_by_id = {str(r.id): r for r in records}

    context_blocks = []
    for i, candidate in enumerate(trace.ranked_candidates):
        record = records_by_id.get(candidate.chunk_id)
        if record is None:
            missing_notes.append(
                f"chunk_id {candidate.chunk_id} (was [{i + 1}] {candidate.filename} p.{candidate.page_number}) "
                "no longer exists in Qdrant -- could not be reconstructed, excluded from the replayed context."
            )
            continue
        text = record.payload["text"]
        context_blocks.append(f"[{i + 1}] (source: {candidate.filename}, page {candidate.page_number})\n{text}")

    if not context_blocks:
        print("Could not reconstruct ANY context chunks -- cannot replay generation. Stopping here.")
        for note in missing_notes:
            print(f"  MISSING: {note}")
        return

    numbered_context = "\n\n".join(context_blocks)

    try:
        generator = _build_generator(trace.generator_provider, trace.generator_model, settings)
    except Exception as exc:
        print(f"Could not construct a live '{trace.generator_provider}' client (missing API key?): {exc}")
        return

    print("--- Original answer ---")
    print(trace.answer)
    print()

    print("--- Replaying (re-running the live model, same prompt/context/temperature)... ---")
    chunks = []
    # prompt_version pins the EXACT historical template (e.g. "v2"), not
    # whatever `_LIVE_VERSIONS` happens to point to today -- this is the
    # whole reason templates are versioned instead of edited in place.
    generation_version = trace.prompt_versions.get("generation")
    async for chunk in generator.stream_answer(trace.question, numbered_context, trace.generator_temperature, generation_version):
        chunks.append(chunk)
    replayed_answer = "".join(chunks)

    print("--- Replayed answer ---")
    print(replayed_answer)
    print()

    if trace.answer.strip() == replayed_answer.strip():
        print("Original and replayed answers match exactly.")
    else:
        print(
            f"Original and replayed answers DIFFER -- expected even with an identical prompt/model/temperature, "
            f"since LLM output isn't perfectly deterministic (temperature={trace.generator_temperature:.1f}). "
            "What matters for replay fidelity is that the INPUT (prompt, context, model, params) was faithfully "
            "reconstructed, not that the output text is byte-identical."
        )

    print("\n--- Fields that could NOT be fully reconstructed ---")
    if missing_notes:
        for note in missing_notes:
            print(f"  - {note}")
    else:
        print("  (none -- every field needed for replay was present in the trace)")

    print(
        "\n--- Known, permanent gap (by design, not a bug) ---\n"
        "  - The judge's raw PRE-verification output isn't stored -- only the post-verification\n"
        "    Judgment (after the literal-substring evidence check) is in the trace. See ChatTrace's\n"
        "    docstring in app/trace/schema.py for why."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_id")
    args = parser.parse_args()
    asyncio.run(main(args.trace_id))
