"""Week 9 (W9-Task-Set-C.md) requirement #5: runs the SAME failing
`search_handbook(as_of=...)` call through the real agent loop against both
the deliberately-worse "before" server and the real "after" server, so the
required before/after transcript shows the MODEL's actual handling of each
error (does it pass the vague "not found" straight through vs. correctly
tell the user no such policy version was ever indexed), not just the raw
tool output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastmcp import Client

from agents.mcp_agent import run_agent
from agents.react_agent import Budgets

_BACKEND_DIR = Path(__file__).resolve().parent.parent

QUESTION = "What does the leave policy say, as of 2000-01-01?"


def _ad_hoc_config(module: str) -> dict:
    return {
        "mcpServers": {
            "policy-search": {
                "command": sys.executable,
                "args": ["-m", f"agents.mcp_servers.{module}"],
                "cwd": str(_BACKEND_DIR),
            }
        }
    }


async def run_error_demo(version: Literal["before", "after"], groq_api_key: str, model: str) -> dict:
    module = "policy_search_server_before" if version == "before" else "policy_search_server"
    client = Client(_ad_hoc_config(module), init_timeout=30.0)
    async with client:
        result = await run_agent(QUESTION, client, groq_api_key, model, Budgets())
    return {
        "question": QUESTION,
        "answer": result.answer,
        "terminated_reason": result.terminated_reason,
        "steps": [
            {"thought": s.thought, "action": s.action, "action_input": s.action_input, "observation": s.observation}
            for s in result.steps
        ],
    }
