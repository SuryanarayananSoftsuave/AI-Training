"""Week 7 extra (not part of the graded rubric): exercises the hybrid
dispatcher (agents/dispatcher.py) against the official 10 race questions
(read-only -- never appended to) plus the multi-employee comparison question
already proven live in agents/branching_case_demo.txt. Confirms the router
sends the 10 to the cheap workflow and the comparison question to the agent,
and writes a separate CSV that never touches the graded agents/race.csv.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/race_dispatcher.py
"""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from app.core.config import get_settings
from app.retrieval.qdrant_store import QdrantStore

from agents.dispatcher import run_dispatch
from agents.employees import EMPLOYEES, expected_notice_days

_QUESTIONS_PATH = Path(__file__).parent.parent / "agents" / "fixtures" / "employee_questions.json"
_DISPATCH_CSV_PATH = Path(__file__).parent.parent / "agents" / "dispatch_race.csv"

_COMPARISON_QUESTION = {
    "id": "q11_comparison",
    "employee_ids": ["E1", "E2"],
    "question": "Compare the notice periods for employee E1 and employee E2 -- which one needs to give more notice?",
}


def _load_official_questions() -> list[dict]:
    return json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def _passed_single(answer: str | None, question: dict) -> bool:
    employee = EMPLOYEES[question["employee_id"]]
    expected = expected_notice_days(employee.jurisdiction, employee.tenure_years)
    return answer is not None and str(expected) in answer


def _passed_comparison(answer: str | None) -> bool:
    expected_days = [str(expected_notice_days(EMPLOYEES[eid].jurisdiction, EMPLOYEES[eid].tenure_years)) for eid in _COMPARISON_QUESTION["employee_ids"]]
    return answer is not None and all(days in answer for days in expected_days)


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    questions = _load_official_questions()

    rows = []
    print(f"Routing {len(questions)} official questions + 1 comparison question...\n")

    for q in questions:
        result = await run_dispatch(q["question"], settings, store, settings.groq_api_key, settings.groq_generator_model)
        passed = _passed_single(result.answer, q)
        status = "OK" if result.system_used == "workflow" else "UNEXPECTED"
        print(f"  [{status}] {q['id']}: routed to {result.system_used} ({result.reason}) -- {'PASS' if passed else 'FAIL'}")
        rows.append({
            "id": q["id"], "question": q["question"], "system_used": result.system_used, "reason": result.reason,
            "passed": passed, "tokens": result.total_tokens, "latency_s": round(result.wall_clock_s, 2),
            "cost_usd": round(result.cost_usd, 6), "fallback_used": result.fallback_used,
        })

    comparison_result = await run_dispatch(_COMPARISON_QUESTION["question"], settings, store, settings.groq_api_key, settings.groq_generator_model)
    comparison_passed = _passed_comparison(comparison_result.answer)
    comparison_status = "OK" if comparison_result.system_used == "agent" else "UNEXPECTED"
    print(f"\n  [{comparison_status}] {_COMPARISON_QUESTION['id']}: routed to {comparison_result.system_used} ({comparison_result.reason}) -- {'PASS' if comparison_passed else 'FAIL'}")
    rows.append({
        "id": _COMPARISON_QUESTION["id"], "question": _COMPARISON_QUESTION["question"], "system_used": comparison_result.system_used,
        "reason": comparison_result.reason, "passed": comparison_passed, "tokens": comparison_result.total_tokens,
        "latency_s": round(comparison_result.wall_clock_s, 2), "cost_usd": round(comparison_result.cost_usd, 6),
        "fallback_used": comparison_result.fallback_used,
    })

    with _DISPATCH_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "system_used", "reason", "passed", "tokens", "latency_s", "cost_usd", "fallback_used"])
        writer.writeheader()
        writer.writerows(rows)

    official_routed_correctly = all(r["system_used"] == "workflow" for r in rows[:-1])
    comparison_routed_correctly = rows[-1]["system_used"] == "agent"
    print(f"\ndispatch_race.csv written to {_DISPATCH_CSV_PATH}")
    print(f"All 10 official questions routed to workflow: {official_routed_correctly}")
    print(f"Comparison question routed to agent: {comparison_routed_correctly}")


if __name__ == "__main__":
    asyncio.run(main())
