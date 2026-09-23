"""Week 6, requirement #1's "2+ regression cases replayed verbatim from
real failed handbook traces": surfaces real candidates from the traces
already on disk, so picking them is a review step, not something typed in
from memory. Two pools, since "failed" means two different things here:

  1. `data/traces/failed_traces.jsonl` -- pipeline-level failures (the
     request itself crashed: a 404/429 from a provider, etc.). Real, but
     usually infra flakiness, not a RAG-quality regression.
  2. `data/traces/successful_traces.jsonl` filtered for a judgment verdict
     of hallucinated/partially_grounded/no_answer, or confidence < 70, or
     (new, found by this script) out_of_scope on a question that reads as
     clearly in-scope -- these are the ones worth a regression case: the
     pipeline ran fine end-to-end but produced a bad answer.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/extract_regression_candidates.py
"""
from __future__ import annotations

from collections import Counter

from app.core.config import get_settings
from app.trace.store import TraceStore

_LOW_CONFIDENCE_THRESHOLD = 70


def main() -> None:
    settings = get_settings()
    store = TraceStore(settings.trace_success_log_path, settings.trace_failure_log_path)

    failures = store.read_failures()
    successes = store.read_successes()

    print(f"=== Pipeline failures ({len(failures)} in {settings.trace_failure_log_path}) ===")
    print("Usually provider/infra flakiness (404/429), not a retrieval/generation quality bug.")
    print("Still eligible as a regression case if the SAME question reliably fails.\n")
    question_error_counts: Counter[str] = Counter(t.question for t in failures)
    for question, count in question_error_counts.most_common():
        marker = "  <-- failed on every attempt" if count == question_error_counts[question] else ""
        example = next(t for t in failures if t.question == question)
        print(f"  [{count}x] {question!r}")
        print(f"         trace_id={example.trace_id}  error={example.error[:100]!r}{marker}")
    print()

    print(f"=== Quality-flagged successes ({len(successes)} total successful traces scanned) ===")
    print("Pipeline completed, but the judgment itself looks bad -- these are the more direct")
    print("evidence of a retrieval/generation regression, not just infra flakiness.\n")

    flagged = [
        t for t in successes
        if t.judgment is not None
        and (
            t.judgment.verdict.value in ("hallucinated", "partially_grounded", "no_answer")
            or t.judgment.confidence < _LOW_CONFIDENCE_THRESHOLD
        )
    ]
    by_verdict: Counter[str] = Counter(t.judgment.verdict.value for t in flagged if t.judgment)
    print(f"{len(flagged)} flagged trace(s), by verdict: {dict(by_verdict)}\n")
    for t in flagged[:30]:
        assert t.judgment is not None
        print(f"  {t.trace_id} | verdict={t.judgment.verdict.value} confidence={t.judgment.confidence} | {t.question!r}")

    # A specific, real pattern this script found in this app's own trace
    # data: the SAME clearly in-scope question landing on out_of_scope
    # across multiple independent runs -- worth surfacing explicitly since
    # it reads as a genuine bug (the off-topic gate or retrieval misfiring
    # on a legitimate question), not just per-run LLM noise.
    out_of_scope_questions = Counter(
        t.question for t in successes if t.judgment is not None and t.judgment.verdict.value == "out_of_scope"
    )
    repeated_out_of_scope = {q: c for q, c in out_of_scope_questions.items() if c >= 2}
    if repeated_out_of_scope:
        print("\n=== Repeated out_of_scope on the same question (worth a closer look) ===")
        for question, count in sorted(repeated_out_of_scope.items(), key=lambda kv: -kv[1]):
            print(f"  [{count}x out_of_scope] {question!r}")
        print(
            "\n  If any of these read as clearly in-scope to you, that's a strong regression-case\n"
            "  candidate: the app is wrongly refusing a question it should answer."
        )

    print(
        "\n--- Next step (yours, not automatable) ---\n"
        "  Pick 2 trace_ids from above whose question+answer you'd genuinely call a regression.\n"
        "  Pull the full record with:\n"
        "    python scripts/replay_trace.py <trace_id>\n"
        "  then hand-copy the question (and, if useful, the expected/correct answer you'd want\n"
        "  instead) into evals/fixtures/week6_eval_cases.json's two REGRESSION_CASE_PLACEHOLDER\n"
        "  entries."
    )


if __name__ == "__main__":
    main()
