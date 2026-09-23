"""The 3 tools shared by both the agent (agents/react_agent.py) and the
fixed workflow (agents/workflow.py) -- same tools, same model, per
requirement #2's "genuinely the same task" bar. Each has exactly one job
and no description overlap with the others (requirement #1/#5):

- `get_employee_record`: employee_id -> known facts about that employee.
  Does not search documents or compute anything.
- `search_handbook`: free-text topic -> policy passage text (real RAG
  retrieval against the already-indexed HR docs, not mocked). Does not
  know about specific employees or compute numbers.
- `get_notice_period_rule` (the new 3rd tool): jurisdiction (enum) +
  tenure_years -> the exact notice-period day count. Does not look up
  employees or search documents -- pure computation over already-known facts.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from app.core.config import Settings
from app.retrieval.pipeline import rerank_candidates, retrieve_candidates
from app.retrieval.qdrant_store import QdrantStore
from agents.employees import EMPLOYEES, Jurisdiction, expected_notice_days

TOOL_DESCRIPTIONS: dict[str, dict] = {
    "get_employee_record": {
        "description": "Look up a specific employee's tenure (years) and jurisdiction by their employee ID. Use this whenever a question names a specific employee and you don't yet know their tenure/jurisdiction. Does not search policy documents and does not compute notice periods.",
        "parameters": {"employee_id": {"type": "string", "description": "e.g. 'E1'"}},
    },
    "search_handbook": {
        "description": "Search the HR policy handbook for general policy text on a topic (e.g. leave types, expense rules, code of conduct). Returns a passage of policy text. Does not know about any specific employee and does not compute numeric entitlements.",
        "parameters": {"query": {"type": "string", "description": "a free-text search query"}},
    },
    "get_notice_period_rule": {
        "description": "Given a jurisdiction and a tenure in years (both already known -- not looked up here), compute the exact notice-period day count that applies, including any long-service bonus. Does not look up employees and does not search documents.",
        "parameters": {
            "jurisdiction": {"type": "string", "enum": ["india", "us", "uk"], "description": "the employee's jurisdiction"},
            "tenure_years": {"type": "number", "description": "the employee's tenure in years"},
        },
    },
}


async def get_employee_record(employee_id: str) -> dict:
    employee = EMPLOYEES.get(employee_id)
    if employee is None:
        return {"error": f"no employee with id {employee_id!r}"}
    return {"employee_id": employee.employee_id, "jurisdiction": employee.jurisdiction, "tenure_years": employee.tenure_years}


async def search_handbook(query: str, settings: Settings, store: QdrantStore) -> dict:
    """Real retrieval against the already-indexed HR policy docs -- reuses
    the exact same retrieve_candidates/rerank_candidates the main /chat
    pipeline uses, not a mocked or hardcoded answer.
    """
    candidates, _total = await retrieve_candidates([query], True, settings.first_stage_limit, None, settings, store)
    if not candidates:
        return {"result": "no matching policy text found"}
    ranked = await rerank_candidates(settings.reranker_model_name, query, candidates, 1)
    point, score = ranked[0]
    return {"result": point.payload["text"][:500], "source": point.payload["filename"], "rerank_score": round(score, 3)}


async def get_notice_period_rule(jurisdiction: Jurisdiction, tenure_years: float) -> dict:
    days = expected_notice_days(jurisdiction, tenure_years)
    return {"jurisdiction": jurisdiction, "tenure_years": tenure_years, "notice_period_days": days}
