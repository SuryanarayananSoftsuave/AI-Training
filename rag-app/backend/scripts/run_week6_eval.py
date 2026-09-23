"""Week 6, the "one command" (requirement #1 / #5 in the submission
checklist): regenerates every case's answer fresh against the live
pipeline, runs the deterministic assertions (evals/assertions.py) plus the
judge, and prints a pass-rate-by-mode table -- never one overall number,
since the rubric explicitly calls out that an average hides a regression
on a hard mode behind an easy mode's good score.

A case PASSES iff every one of its `checks` passes AND (when a real judge
call happened for it) the judge's verdict is "grounded". A case whose
answer never reached the real judge (the no-candidates/off-topic gates
short-circuited with their own fixed verdict -- see
evals/week6_eval.py::generate_raw_answer's docstring) is graded on that
gate's verdict instead of calling the judge a second time for no reason.

Independent of the one-time blind-labeling bootstrap
(scripts/label_answers.py) -- this script regenerates fresh answers every
run (so it reflects current app behavior, not a frozen snapshot) and is
meant to be re-run any time you change chunking/retrieval/prompts/models,
to see what moved.

Prerequisite: the 8 HR policy docs must already be uploaded and `indexed`
in the running Qdrant instance.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/run_week6_eval.py
    python scripts/run_week6_eval.py --provider groq   # route around a stuck/rate-limited Gemini
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict

from app.core.config import get_settings
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.retrieval.qdrant_store import QdrantStore
from evals.week6_eval import RESULTS_PATH, grade_one, load_cases


async def _grade_one(case: dict, settings, store, generator, provider: str, i: int, total: int) -> dict:
    import time

    t0 = time.monotonic()
    print(f"[{i}/{total}] {case['id']}: generating + grading...", flush=True)
    result = await grade_one(case, settings, store, generator, provider)
    print(
        f"[{i}/{total}] {case['id']}: graded ({time.monotonic() - t0:.1f}s) -> "
        f"{result['answer'][:80]!r} judge={result['judge_verdict']}({result['judged_by']})",
        flush=True,
    )
    return result


def _build_client(provider: str, settings):
    if provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_generator_model, settings.groq_judge_model)
    return GeminiClient(settings.gemini_api_key, settings.gemini_generator_model, settings.gemini_judge_model)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    args = parser.parse_args()

    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    generator = _build_client(args.provider, settings)

    cases = load_cases()
    print(f"Running Week 6 eval over {len(cases)} case(s), provider={args.provider} ...\n")

    results = []
    for i, case in enumerate(cases, 1):
        result = await _grade_one(case, settings, store, generator, args.provider, i, len(cases))
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{i}/{len(cases)}] {status}  {case['id']} ({case['mode']})  judge={result['judge_verdict']}({result['judged_by']})")
        for name, detail in result["checks"].items():
            mark = "OK" if detail["passed"] else "XX"
            print(f"         [{mark}] {name}: {detail['reason']}")
        results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    by_mode: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r["passed"])

    print("\n=== Pass rate by mode ===")
    for mode, outcomes in sorted(by_mode.items()):
        rate = sum(outcomes) / len(outcomes)
        print(f"  {mode:<40} {sum(outcomes)}/{len(outcomes)}  ({rate:.0%})")

    overall = sum(r["passed"] for r in results) / len(results)
    print(f"\nOverall: {sum(r['passed'] for r in results)}/{len(results)} ({overall:.0%})")
    print(f"\nAssertion vs judge-graded criteria in this run: {len(set().union(*(r['checks'] for r in results)) if results else set())} assertion type(s) vs 1 judge-graded criterion.")
    print(f"Full results written to {RESULTS_PATH}.")


if __name__ == "__main__":
    asyncio.run(main())
