"""Week 7 requirement #2: the identical task re-implemented as a FIXED
workflow -- same tools (agents/tools.py), same model, same output
contract (one sentence naming the notice-period day count), same raw-text
input as the agent gets. No loop: the step order is hardcoded in Python,
not decided by an LLM call.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.retrieval.qdrant_store import QdrantStore
from groq import AsyncGroq

from agents.react_agent import PLACEHOLDER_COST_PER_1K_TOKENS
from agents.tools import get_employee_record, get_notice_period_rule

_EMPLOYEE_ID_RE = re.compile(r"\bE\d+\b")


@dataclass
class WorkflowResult:
    question: str
    answer: str | None
    total_tokens: int
    cost_usd: float
    wall_clock_s: float
    error: str | None = None


async def run_workflow(
    question: str, settings: Settings, store: QdrantStore, groq_api_key: str, model: str,
) -> WorkflowResult:
    started = time.monotonic()

    # Step 1 (hardcoded): parse the employee ID out of the raw question --
    # same raw-text input the agent gets, no pre-extracted structured input.
    match = _EMPLOYEE_ID_RE.search(question)
    if not match:
        return WorkflowResult(question, None, 0, 0.0, time.monotonic() - started, error="no employee id found in question")
    employee_id = match.group(0)

    # Step 2 (hardcoded): same tool the agent has, called unconditionally.
    record = await get_employee_record(employee_id)
    if "error" in record:
        return WorkflowResult(question, None, 0, 0.0, time.monotonic() - started, error=record["error"])

    # Step 3 (hardcoded, THE branch requirement #3 needs -- but the branch
    # is inside get_notice_period_rule's own logic, not a workflow-level
    # if/else, since "no loop" doesn't mean "no business logic": the
    # workflow still calls the same tool with the same tenure-dependent
    # rule the agent would discover via reasoning).
    rule = await get_notice_period_rule(record["jurisdiction"], record["tenure_years"])

    # Step 4 (hardcoded, one LLM call): phrase the final answer -- same
    # model as the agent, same output contract (one sentence naming the
    # exact day count).
    client = AsyncGroq(api_key=groq_api_key)
    prompt = (
        f"Question: {question}\n"
        f"Known facts: employee {employee_id} is in jurisdiction {record['jurisdiction']}, "
        f"tenure {record['tenure_years']} years, notice period is {rule['notice_period_days']} days.\n"
        "Write ONE sentence answering the question, stating the exact notice period in days."
    )
    response = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
    )
    total_tokens = response.usage.total_tokens
    elapsed = time.monotonic() - started
    cost = total_tokens / 1000 * PLACEHOLDER_COST_PER_1K_TOKENS
    return WorkflowResult(question, response.choices[0].message.content, total_tokens, cost, elapsed)
