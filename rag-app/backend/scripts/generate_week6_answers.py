"""Week 6, step 1: generate the raw answer for every eval case via the
REAL pipeline (retrieve -> rerank -> generate), with the judge deliberately
never called (`skip_judge=True`). This has to run and be saved BEFORE any
blind labeling -- labeling against an answer the judge already touched
isn't blind, it's "agreeing with yourself with extra steps" (the rubric's
own words for this exact mistake).

Prerequisite: the 8 HR policy docs must already be uploaded and `indexed`
in the running Qdrant instance (same as scripts/run_hit_rate_eval.py).

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/generate_week6_answers.py
"""
from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.llm.gemini_client import GeminiClient
from app.retrieval.qdrant_store import QdrantStore
from evals.week6_eval import generate_raw_answer, load_cases, save_raw_answers


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    generator = GeminiClient(settings.gemini_api_key, settings.gemini_generator_model, settings.gemini_judge_model)

    cases = load_cases()
    print(f"Generating raw answers for {len(cases)} case(s) (judge NOT called) ...\n")

    answers = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']!r}")
        answer = await generate_raw_answer(case, settings, store, generator, "gemini")
        answers.append(answer)
        print(f"         -> {answer['answer'][:100]!r}...")

    save_raw_answers(answers)
    print(f"\nSaved {len(answers)} raw answer(s) to evals/week6_raw_answers.json.")
    print("Next: python scripts/label_answers.py")


if __name__ == "__main__":
    asyncio.run(main())
