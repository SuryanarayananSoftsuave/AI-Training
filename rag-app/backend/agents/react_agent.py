"""Week 7 (W7-Task-Set-C.md): a hand-built ReAct-style tool-calling loop.
Not Groq/Gemini native function-calling -- a manually parsed structured
JSON step (thought/action/action_input), so budget enforcement and
per-lap token counting are fully auditable in this code rather than
hidden inside an SDK's function-calling machinery.

Requirement #4 (budgets enforced in code, not just declared): every lap
checks max_iterations, max_tokens (summed across ALL laps, not just the
final call -- the rubric's own "common mistakes" section calls out
under-counting this), max_cost, and wall-clock, and returns a clean
`BudgetExceeded` result instead of continuing.

Cost: PLACEHOLDER_COST_PER_1K_TOKENS below is NOT verified real Groq
pricing (not fetched from a live pricing page this session) -- it exists
so the four required numbers are all computed consistently between the
agent and the workflow for comparison purposes. Treat the dollar figures
as illustrative/relative, not a real billing claim, until checked against
console.groq.com's current pricing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore
from groq import AsyncGroq

from agents.conversation import ConversationState, render_conversation_block
from agents.tools import TOOL_DESCRIPTIONS, get_employee_record, get_notice_period_rule, search_handbook

PLACEHOLDER_COST_PER_1K_TOKENS = 0.0002  # NOT verified real pricing -- see module docstring


@dataclass
class Budgets:
    max_iterations: int = 6
    max_tokens: int = 4000
    max_cost_usd: float = 0.01
    max_wall_clock_s: float = 30.0


@dataclass
class StepRecord:
    thought: str
    action: str
    action_input: dict
    observation: str


@dataclass
class AgentResult:
    question: str
    answer: str | None
    iterations: int
    total_tokens: int
    cost_usd: float
    wall_clock_s: float
    steps: list[StepRecord] = field(default_factory=list)
    terminated_reason: str | None = None  # None = finished normally; else which budget fired


_STEP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_step",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": ["get_employee_record", "search_handbook", "get_notice_period_rule", "final_answer"]},
                "employee_id": {"type": ["string", "null"]},
                "query": {"type": ["string", "null"]},
                "jurisdiction": {"type": ["string", "null"], "enum": ["india", "us", "uk", None]},
                "tenure_years": {"type": ["number", "null"]},
                "final_answer": {"type": ["string", "null"]},
            },
            "required": ["thought", "action", "employee_id", "query", "jurisdiction", "tenure_years", "final_answer"],
            "additionalProperties": False,
        },
    },
}


def _tools_block() -> str:
    lines = []
    for name, spec in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}({', '.join(spec['parameters'])}): {spec['description']}")
    return "\n".join(lines)


def _build_prompt(question: str, steps: list[StepRecord], conversation: ConversationState | None = None) -> str:
    history = "\n".join(
        f"Step {i + 1}:\n  Thought: {s.thought}\n  Action: {s.action}({s.action_input})\n  Observation: {s.observation}"
        for i, s in enumerate(steps)
    )
    return f"""You are an HR agent answering an employee's question by calling tools, one at a time.

Available tools:
{_tools_block()}

{render_conversation_block(conversation)}Question: {question}

{"Steps so far:\n" + history if steps else "(no steps taken yet)"}

Decide the SINGLE next step. If you have enough information to answer, set action="final_answer" and
put the complete answer text in final_answer (include the specific number/fact). Otherwise pick exactly
one tool and fill in only the fields it needs (leave the rest null)."""


async def _call_step(
    client: AsyncGroq, model: str, question: str, steps: list[StepRecord], conversation: ConversationState | None = None,
) -> tuple[dict, int]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _build_prompt(question, steps, conversation)}],
        response_format=_STEP_SCHEMA,
        temperature=0.0,
    )
    data = json.loads(response.choices[0].message.content or "{}")
    return data, response.usage.total_tokens


async def _execute_tool(
    action: str, data: dict, settings: Settings, store: QdrantStore,
    conversation: ConversationState | None = None, session_store: SessionStore | None = None,
) -> tuple[dict, str]:
    if action == "get_employee_record":
        result = await get_employee_record(data["employee_id"])
        if "error" not in result and conversation is not None:
            conversation.employee_id = result["employee_id"]
            conversation.jurisdiction = result["jurisdiction"]
            if session_store is not None:
                session_store.set_jurisdiction(conversation.session_id, result["employee_id"], result["jurisdiction"])
        return {"employee_id": data["employee_id"]}, json.dumps(result)
    if action == "search_handbook":
        result = await search_handbook(data["query"], settings, store)
        return {"query": data["query"]}, json.dumps(result)
    if action == "get_notice_period_rule":
        result = await get_notice_period_rule(data["jurisdiction"], data["tenure_years"])
        return {"jurisdiction": data["jurisdiction"], "tenure_years": data["tenure_years"]}, json.dumps(result)
    return {}, f"unknown action {action!r}"


async def run_agent(
    question: str, settings: Settings, store: QdrantStore, groq_api_key: str, model: str, budgets: Budgets,
    conversation: ConversationState | None = None, session_store: SessionStore | None = None,
) -> AgentResult:
    # max_retries above the SDK's default of 2: a multi-turn conversation
    # (agents/conversation.py) can rack up enough calls in quick succession
    # to trip Groq's per-minute output-token rate limit even when each
    # individual call is small -- the SDK already backs off using the
    # server's Retry-After header, it just needs more attempts budgeted.
    client = AsyncGroq(api_key=groq_api_key, max_retries=6)
    steps: list[StepRecord] = []
    total_tokens = 0
    started = time.monotonic()
    iteration = 0

    while True:
        iteration += 1
        elapsed = time.monotonic() - started
        cost_so_far = total_tokens / 1000 * PLACEHOLDER_COST_PER_1K_TOKENS

        if iteration > budgets.max_iterations:
            return AgentResult(question, None, iteration - 1, total_tokens, cost_so_far, elapsed, steps, "max_iterations")
        if total_tokens > budgets.max_tokens:
            return AgentResult(question, None, iteration - 1, total_tokens, cost_so_far, elapsed, steps, "max_tokens")
        if cost_so_far > budgets.max_cost_usd:
            return AgentResult(question, None, iteration - 1, total_tokens, cost_so_far, elapsed, steps, "max_cost")
        if elapsed > budgets.max_wall_clock_s:
            return AgentResult(question, None, iteration - 1, total_tokens, cost_so_far, elapsed, steps, "max_wall_clock")

        data, step_tokens = await _call_step(client, model, question, steps, conversation)
        total_tokens += step_tokens

        if data.get("action") == "final_answer":
            elapsed = time.monotonic() - started
            cost = total_tokens / 1000 * PLACEHOLDER_COST_PER_1K_TOKENS
            return AgentResult(question, data.get("final_answer", ""), iteration, total_tokens, cost, elapsed, steps, None)

        action_input, observation = await _execute_tool(data["action"], data, settings, store, conversation, session_store)
        steps.append(StepRecord(thought=data.get("thought", ""), action=data["action"], action_input=action_input, observation=observation))
