"""Week 8 (W8-Task-Set-C.md): runs the 10 trajectory cases through the real
agent twice -- once with react_agent.py's require_employee_lookup gate off
("before"), once with it on ("after") -- and prints the same numbers the
/agents/trajectory/run API route streams. Writes
agents/trajectory_results_before.json / _after.json, exactly what a
UI-triggered run also produces (agents/trajectory_eval.py::run_case is the
one shared implementation).

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/run_trajectory_eval.py
"""
from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore

from agents import trajectory_eval


def _print_case(result: trajectory_eval.CaseResult) -> None:
    if result.error:
        print(f"  {result.id}: ERRORED -- {result.error}")
        return
    tool_choice = "PASS" if result.tool_choice_passed else "FAIL"
    outcome = "n/a" if result.outcome_passed is None else ("PASS" if result.outcome_passed else "FAIL")
    modes = f" modes={result.zoo_modes}" if result.zoo_modes else ""
    print(
        f"  {result.id}: tool_choice={tool_choice} outcome={outcome} "
        f"args={result.argument_valid}/{result.argument_checked} "
        f"steps={len(result.steps)} tokens={result.total_tokens} cost=${result.cost_usd:.6f}{modes}"
    )


def _print_report(label: str, report: dict) -> None:
    print(f"\n=== {label} ===")
    if report["n_errored"]:
        print(f"⚠ {report['n_errored']}/{report['n_cases']} case(s) errored -- excluded from every number below.")
    print(f"tool_choice_accuracy:    {report['tool_choice_accuracy']}%")
    print(f"argument_validity_rate:  {report['argument_validity_rate']}% ({report['argument_checks_total']} checks)")
    print(f"step_efficiency_avg:     {report['step_efficiency_avg']}")
    print(f"cost_p50 / cost_max:     ${report['cost_p50_usd']:.6f} / ${report['cost_max_usd']:.6f}")
    print(f"outcome_pass_rate:       {report['outcome_pass_rate']}%")
    print(f"gap (outcome - trajectory): {report['gap']}")
    if report["named_right_answer_wrong_path_example"]:
        ex = report["named_right_answer_wrong_path_example"]
        print(f"right-answer-wrong-path example: {ex['id']} -- {ex['tool_choice_detail']}")
    print(f"mode_counts: {report['mode_counts']}")


async def _run_phase(mitigation: bool, settings, store, session_store) -> dict:
    label = "AFTER (mitigation on)" if mitigation else "BEFORE (mitigation off)"
    print(f"\nRunning {label}...")
    results = []
    for case in trajectory_eval.load_cases():
        result = await trajectory_eval.run_case(
            case, settings, store, settings.groq_api_key, settings.groq_generator_model, mitigation, session_store,
        )
        _print_case(result)
        results.append(result)
    report = trajectory_eval.build_report(results)
    trajectory_eval.save_report(report, "after" if mitigation else "before")
    _print_report(label, report)
    return report


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    session_store = SessionStore(settings.session_store_path)

    before_report = await _run_phase(False, settings, store, session_store)
    after_report = await _run_phase(True, settings, store, session_store)

    diff = trajectory_eval.regression_diff(before_report, after_report)
    print("\n=== Regression check (every mode, before -> after) ===")
    for row in diff["rows"]:
        print(f"  {row['mode']}: {row['before']} -> {row['after']}  ({row['flag']})")

    print(
        f"\nResults written to {trajectory_eval.results_path('before')} and "
        f"{trajectory_eval.results_path('after')} (read by the frontend's Trajectory Explorer panel)."
    )


if __name__ == "__main__":
    asyncio.run(main())
