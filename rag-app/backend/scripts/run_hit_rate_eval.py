"""Week 4 deliverable: hit-rate@3, baseline (no MMR) vs. with MMR, over the
same labeled question set (`evals/fixtures/hit_rate_questions.json`) -- the
before/after number the mentor review asks for, plus which questions the
change did and didn't fix.

Prerequisite: the 8 HR policy PDFs in `sample_data/hr_policy_kb/` must
already be uploaded and `indexed` in the running Qdrant instance.

Run from backend/, with PYTHONPATH set to it (same convention the app
itself uses):

    cd backend
    $env:PYTHONPATH = "$PWD"          # PowerShell
    python scripts/run_hit_rate_eval.py
"""
from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.retrieval.qdrant_store import QdrantStore
from evals.hit_rate import compute_hit_rate_at_k, load_questions

K = 3


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    questions = load_questions()

    print(f"Loaded {len(questions)} labeled questions from evals/fixtures/hit_rate_questions.json.\n")

    baseline_rate, baseline_detail = await compute_hit_rate_at_k(questions, K, use_mmr=False, settings=settings, store=store)
    mmr_rate, mmr_detail = await compute_hit_rate_at_k(questions, K, use_mmr=True, settings=settings, store=store)

    baseline_hits = sum(d["hit"] for d in baseline_detail)
    mmr_hits = sum(d["hit"] for d in mmr_detail)

    print(f"Baseline hit-rate@{K} (no MMR): {baseline_rate:.2%}  ({baseline_hits}/{len(questions)})")
    print(f"With MMR hit-rate@{K}:         {mmr_rate:.2%}  ({mmr_hits}/{len(questions)})")
    print()

    fixed = [m["id"] for b, m in zip(baseline_detail, mmr_detail) if not b["hit"] and m["hit"]]
    regressed = [m["id"] for b, m in zip(baseline_detail, mmr_detail) if b["hit"] and not m["hit"]]
    still_missed = [m["id"] for m in mmr_detail if not m["hit"]]

    if fixed:
        print(f"Fixed by MMR:              {fixed}")
    if regressed:
        print(f"Regressed by MMR:          {regressed}")
    if still_missed:
        print(f"Still missed (not fixed):  {still_missed}")
    if not (fixed or regressed):
        print("No change between baseline and MMR on this question set.")

    print("\nPer-question detail:")
    print(f"{'id':<5} {'baseline':<10} {'mmr':<10} question")
    for b, m in zip(baseline_detail, mmr_detail):
        print(f"{b['id']:<5} {'HIT' if b['hit'] else 'MISS':<10} {'HIT' if m['hit'] else 'MISS':<10} {b['question']}")
        if not b["hit"] or not m["hit"]:
            print(f"      expected={b['expected']} baseline_top{K}={b['actual_top_k']} mmr_top{K}={m['actual_top_k']}")


if __name__ == "__main__":
    asyncio.run(main())
