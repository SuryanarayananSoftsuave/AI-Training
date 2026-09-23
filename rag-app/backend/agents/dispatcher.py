"""Week 7 extra (not part of the graded rubric): a hybrid router that picks
workflow-vs-agent per question instead of racing both globally. Wraps
run_workflow/run_agent unchanged -- workflow.py is not touched by this file.

The routing rule is the exact decision rule proven live in
agents/branching_case_demo.txt: a question naming exactly one employee, with
no comparison/conditional language, never makes the agent's tool-call path
diverge from the workflow's hard-coded two-step sequence -- so send it to
the cheap workflow. Anything else (0 ids -- the workflow can't even start;
2+ ids; or comparison/conditional phrasing) genuinely needs the agent to
decide how many lookups to do. This check is a regex + keyword list, not an
LLM call, on purpose: spending a model call just to decide which system to
use would defeat the point of routing to the cheap path at all.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from app.core.config import Settings
from app.retrieval.qdrant_store import QdrantStore

from agents.react_agent import Budgets, StepRecord, run_agent
from agents.workflow import run_workflow

# Deliberately a separate copy of workflow.py's pattern (not imported from
# there) so this file never has to reach into another module's
# underscore-prefixed constant, and workflow.py stays completely untouched.
_EMPLOYEE_ID_RE = re.compile(r"\bE\d+\b")

_COMPARISON_KEYWORDS = ("compare", "versus", " vs ", "vs.", "which one", "both", "difference between", "more than", "less than")
# Deliberately narrow phrases, not a bare "if " -- "if they resign" appears
# in several of the official single-lookup questions and isn't a real
# conditional branch; "also check"/"in case" only show up when a SECOND
# tool call is genuinely contingent on the first one's result.
_CONDITIONAL_KEYWORDS = ("also check", "in case")

System = Literal["workflow", "agent"]


@dataclass
class DispatchResult:
    question: str
    system_used: System
    reason: str
    answer: str | None
    total_tokens: int
    cost_usd: float
    wall_clock_s: float
    fallback_used: bool = False
    error: str | None = None
    agent_steps: list[StepRecord] = field(default_factory=list)
    terminated_reason: str | None = None


def choose_system(question: str) -> tuple[System, str]:
    ids = sorted(set(_EMPLOYEE_ID_RE.findall(question)))
    lowered = question.lower()
    keyword_hit = next((kw for kw in (*_COMPARISON_KEYWORDS, *_CONDITIONAL_KEYWORDS) if kw in lowered), None)

    if len(ids) == 1 and keyword_hit is None:
        return "workflow", f"1 employee id ({ids[0]}), no comparison/conditional language"
    if len(ids) == 0:
        return "agent", "no employee id found -- the fixed workflow cannot even start"
    if keyword_hit is not None:
        return "agent", f"comparison/conditional language detected ({keyword_hit!r})"
    return "agent", f"{len(ids)} employee ids ({', '.join(ids)}) -- path depends on how many to look up"


async def run_dispatch(
    question: str, settings: Settings, store: QdrantStore, groq_api_key: str, model: str, budgets: Budgets | None = None,
) -> DispatchResult:
    system, reason = choose_system(question)

    if system == "workflow":
        started = time.monotonic()
        result = await run_workflow(question, settings, store, groq_api_key, model)
        if result.error is None:
            return DispatchResult(
                question, "workflow", reason, result.answer, result.total_tokens, result.cost_usd, result.wall_clock_s,
            )
        # Workflow's own hard-coded path couldn't resolve this input (e.g. an
        # unknown employee id) -- fall back to the agent rather than
        # surfacing a dead end, and be explicit that a fallback happened.
        agent_result = await run_agent(question, settings, store, groq_api_key, model, budgets or Budgets())
        fallback_wall_clock = time.monotonic() - started
        return DispatchResult(
            question, "agent", f"{reason}; fallback after workflow error: {result.error}", agent_result.answer,
            agent_result.total_tokens, agent_result.cost_usd, fallback_wall_clock, fallback_used=True,
            error=result.error, agent_steps=agent_result.steps, terminated_reason=agent_result.terminated_reason,
        )

    agent_result = await run_agent(question, settings, store, groq_api_key, model, budgets or Budgets())
    return DispatchResult(
        question, "agent", reason, agent_result.answer, agent_result.total_tokens, agent_result.cost_usd,
        agent_result.wall_clock_s, agent_steps=agent_result.steps, terminated_reason=agent_result.terminated_reason,
    )
