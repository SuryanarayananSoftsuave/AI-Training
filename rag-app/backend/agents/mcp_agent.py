"""Week 9 (W9-Task-Set-C.md) base build: the agent that discovers its tools
over MCP instead of having them wired in by hand.

This file must NEVER name search_handbook / get_grade_band /
get_accrued_leave_balance as Python identifiers or constants -- only ever
the strings `tools/list` returns at runtime, via agents/mcp_config.py's
config. That is the entire mechanism behind requirement #2's "zero lines
changed" proof: adding the "hris" block to mcp_config_two_servers.json and
reconnecting is the only thing that ever changes what this file's loop can
do -- see agents/mcp_diff.py and agents/mcp_agent_baseline.py.

Reuses agents/react_agent.py's Budgets/StepRecord/AgentResult dataclasses
(pure data, no MCP-specific logic) so frontend/agent_view.py's
render_step_trace can render this agent's trace unmodified, and drives
Groq's NATIVE tools=[...] function-calling -- built at runtime from whatever
tools/list returns -- rather than react_agent.py's hand-rolled fixed-enum
JSON-step schema, since a fixed enum of allowed actions would itself be a
form of hard-coding incompatible with this week's entire point.
"""
from __future__ import annotations

import json
import time
from typing import Literal

from fastmcp import Client
from groq import AsyncGroq

from agents.mcp_config import load_config
from agents.react_agent import PLACEHOLDER_COST_PER_1K_TOKENS, AgentResult, Budgets, StepRecord


def build_client(label: Literal["one", "two"]) -> Client:
    # init_timeout only bounds the initialize/tools-list handshake, which is
    # now fast (model loading is lazy, inside search_handbook itself) --
    # generous mainly as a safety margin on a loaded dev machine.
    return Client(load_config(label), init_timeout=30.0)


async def discover_tools(client: Client) -> list[dict]:
    """Must be called with `client` already connected (inside `async with`).
    This IS the live tools/list result -- the UI's tool table and the tool
    count report are built directly from this, never from notes.
    """
    tools = await client.list_tools()
    return [{"name": t.name, "description": t.description} for t in tools]


async def _tool_specs(client: Client) -> list[dict]:
    tools = await client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _stringify_result(result) -> str:
    if result.is_error:
        return result.content[0].text if result.content else "(no error detail)"
    if result.data is not None:
        try:
            return json.dumps(result.data, default=str)
        except TypeError:
            pass
    if result.structured_content is not None:
        return json.dumps(result.structured_content, default=str)
    return "\n".join(getattr(block, "text", str(block)) for block in result.content)


async def run_agent(
    question: str, client: Client, groq_api_key: str, model: str, budgets: Budgets,
) -> AgentResult:
    """Must be called with `client` already connected. Drives Groq's native
    function-calling; accepted tool calls are dispatched through `client`,
    which routes to whichever underlying server owns that (possibly
    fastmcp-prefixed) tool name -- this loop never knows or cares which.
    """
    groq_client = AsyncGroq(api_key=groq_api_key, max_retries=6)
    tools = await _tool_specs(client)
    messages: list[dict] = [{"role": "user", "content": question}]
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

        response = await groq_client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.0,
        )
        total_tokens += response.usage.total_tokens
        message = response.choices[0].message

        if not message.tool_calls:
            elapsed = time.monotonic() - started
            cost = total_tokens / 1000 * PLACEHOLDER_COST_PER_1K_TOKENS
            return AgentResult(question, message.content, iteration, total_tokens, cost, elapsed, steps, None)

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            try:
                result = await client.call_tool(tool_call.function.name, args, raise_on_error=False)
                observation = _stringify_result(result)
            except Exception as exc:  # transport/protocol failure, not a tool-side error
                observation = f"error calling {tool_call.function.name}: {exc}"
            steps.append(StepRecord(
                thought=message.content or "",
                action=tool_call.function.name,
                action_input=args,
                observation=observation,
            ))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})
