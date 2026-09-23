"""Week 7 (W7-Task-Set-C.md): races the ReAct agent against the fixed
workflow over the same 10 employee questions, reports the 4 required
numbers per system (pass rate, p50 latency, total tokens, cost/question),
writes race.csv, and separately demonstrates one clean budget termination
(requirement #4) by re-running one question with a deliberately tight
max_iterations budget.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/race_agent_vs_workflow.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import statistics
from pathlib import Path

from app.core.config import get_settings
from app.retrieval.qdrant_store import QdrantStore

from agents.employees import EMPLOYEES, expected_notice_days
from agents.react_agent import Budgets, run_agent
from agents.workflow import run_workflow

_QUESTIONS_PATH = Path(__file__).parent.parent / "agents" / "fixtures" / "employee_questions.json"
_RACE_CSV_PATH = Path(__file__).parent.parent / "agents" / "race.csv"
_BUDGET_LOG_PATH = Path(__file__).parent.parent / "agents" / "budget_termination_log.txt"


def _load_questions() -> list[dict]:
    return json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def _passed(answer: str | None, expected_days: int) -> bool:
    return answer is not None and str(expected_days) in answer


async def _run_agent_side(questions: list[dict], settings, store) -> list[dict]:
    results = []
    for q in questions:
        employee = EMPLOYEES[q["employee_id"]]
        expected = expected_notice_days(employee.jurisdiction, employee.tenure_years)
        r = await run_agent(q["question"], settings, store, settings.groq_api_key, settings.groq_generator_model, Budgets())
        passed = _passed(r.answer, expected) if r.terminated_reason is None else False
        print(f"  [agent] {q['id']}: {'PASS' if passed else 'FAIL'} ({r.iterations} laps, {r.total_tokens} tok, {r.wall_clock_s:.1f}s) -> {r.answer!r}")
        results.append({
            "id": q["id"], "passed": passed, "latency_s": r.wall_clock_s, "tokens": r.total_tokens,
            "cost_usd": r.cost_usd, "terminated_reason": r.terminated_reason,
        })
    return results


async def _run_workflow_side(questions: list[dict], settings, store) -> list[dict]:
    results = []
    for q in questions:
        employee = EMPLOYEES[q["employee_id"]]
        expected = expected_notice_days(employee.jurisdiction, employee.tenure_years)
        r = await run_workflow(q["question"], settings, store, settings.groq_api_key, settings.groq_generator_model)
        passed = _passed(r.answer, expected)
        print(f"  [workflow] {q['id']}: {'PASS' if passed else 'FAIL'} ({r.tokens if hasattr(r,'tokens') else r.total_tokens} tok, {r.wall_clock_s:.1f}s) -> {r.answer!r}")
        results.append({"id": q["id"], "passed": passed, "latency_s": r.wall_clock_s, "tokens": r.total_tokens, "cost_usd": r.cost_usd})
    return results


def _summarize(name: str, results: list[dict]) -> dict:
    n = len(results)
    pass_rate = sum(r["passed"] for r in results) / n
    p50_latency = statistics.median(r["latency_s"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    cost_per_q = sum(r["cost_usd"] for r in results) / n
    print(f"\n{name}: pass_rate={pass_rate:.0%}  p50_latency={p50_latency:.2f}s  total_tokens={total_tokens}  cost/question=${cost_per_q:.5f}")
    return {"system": name, "pass_rate": pass_rate, "p50_latency_s": round(p50_latency, 2), "total_tokens": total_tokens, "cost_per_question_usd": round(cost_per_q, 6)}


async def _demo_budget_termination(settings, store) -> None:
    """Requirement #4: a log of one run that hits a budget and terminates
    cleanly. Uses a deliberately tight max_iterations=1 on a question that
    genuinely needs 2+ tool calls (employee lookup, then the notice-period
    rule) -- a real termination, not a fabricated log line.
    """
    tight_budgets = Budgets(max_iterations=1, max_tokens=4000, max_cost_usd=0.01, max_wall_clock_s=30.0)
    question = "What is the notice period for employee E1 if they resign?"
    r = await run_agent(question, settings, store, settings.groq_api_key, settings.groq_generator_model, tight_budgets)
    log_lines = [
        f"Question: {question}",
        f"Budgets: max_iterations={tight_budgets.max_iterations}, max_tokens={tight_budgets.max_tokens}, "
        f"max_cost_usd={tight_budgets.max_cost_usd}, max_wall_clock_s={tight_budgets.max_wall_clock_s}",
        f"Terminated: reason={r.terminated_reason}  after {r.iterations} completed lap(s), {r.total_tokens} tokens, {r.wall_clock_s:.2f}s",
        f"Answer produced: {r.answer!r} (None = budget fired before a final_answer step)",
        "Steps taken before termination:",
    ]
    for i, s in enumerate(r.steps, 1):
        log_lines.append(f"  {i}. thought={s.thought!r} action={s.action}({s.action_input}) -> observation={s.observation}")
    log_text = "\n".join(log_lines)
    _BUDGET_LOG_PATH.write_text(log_text, encoding="utf-8")
    print(f"\n=== Budget termination demo ===\n{log_text}\nWritten to {_BUDGET_LOG_PATH}")


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    questions = _load_questions()

    print(f"Racing agent vs workflow over {len(questions)} questions...\n")

    print("--- Agent ---")
    agent_results = await _run_agent_side(questions, settings, store)
    print("\n--- Workflow ---")
    workflow_results = await _run_workflow_side(questions, settings, store)

    agent_summary = _summarize("AGENT", agent_results)
    workflow_summary = _summarize("WORKFLOW", workflow_results)

    with _RACE_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "pass_rate", "p50_latency_s", "total_tokens", "cost_per_question_usd"])
        writer.writeheader()
        writer.writerow(agent_summary)
        writer.writerow(workflow_summary)
    print(f"\nrace.csv written to {_RACE_CSV_PATH}")

    await _demo_budget_termination(settings, store)


if __name__ == "__main__":
    asyncio.run(main())
