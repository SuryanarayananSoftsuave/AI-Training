"""Week 9 (W9-Task-Set-C.md) backend surface: the FastAPI endpoints backing
the MCP Discovery/Wire Capture/Recoverable Errors/Ask/Risk Note UI panels,
plus one internal endpoint the policy-search MCP server subprocess itself
calls back into (see its module docstring for why: loading a second copy of
the embedder/reranker in that subprocess hit a real Windows pagefile crash
on a dev machine already running the full stack, so it reuses the models
THIS process already has loaded instead of loading its own).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.agents import StepOut
from app.core.config import Settings, get_settings
from app.core.dependencies import get_store
from app.retrieval.pipeline import rerank_candidates, retrieve_candidates
from app.retrieval.qdrant_store import QdrantStore

from agents import mcp_agent, mcp_diff, mcp_error_demo, mcp_wire_capture
from agents.react_agent import Budgets

router = APIRouter(prefix="/mcp", tags=["mcp"])

# backend/app/api/mcp.py -> parent.parent.parent == backend/, same convention
# app/api/agents.py already uses for its own _AGENTS_DIR.
_AGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "agents"
_ERROR_DEMO_STATE_PATH = _AGENTS_DIR / "_error_demo_state.json"


class InternalSearchHandbookRequest(BaseModel):
    query: str


class InternalSearchHandbookResponse(BaseModel):
    result: str
    source: str | None = None
    section_heading: str | None = None
    rerank_score: float | None = None


@router.post("/internal/search-handbook", response_model=InternalSearchHandbookResponse)
async def internal_search_handbook(
    request: InternalSearchHandbookRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> InternalSearchHandbookResponse:
    """Not part of the public API surface -- called only by
    agents/mcp_servers/policy_search_server(_before).py, over loopback HTTP,
    so the actual retrieve_candidates/rerank_candidates call runs against
    THIS process's already-loaded embedder/reranker/QdrantStore rather than
    a second copy in the MCP server's own subprocess.
    """
    candidates, _total = await retrieve_candidates(
        [request.query], True, settings.first_stage_limit, None, settings, store
    )
    if not candidates:
        return InternalSearchHandbookResponse(result="no matching policy text found")
    ranked = await rerank_candidates(settings.reranker_model_name, request.query, candidates, 1)
    point, score = ranked[0]
    return InternalSearchHandbookResponse(
        result=point.payload["text"][:500],
        source=point.payload["filename"],
        section_heading=point.payload.get("section_heading"),
        rerank_score=round(score, 3),
    )


# --- Discovery (base-build checklist items 1-2; requirement #3, 15 pts) ----


class McpToolOut(BaseModel):
    name: str
    description: str | None


class McpToolsResponse(BaseModel):
    config_label: Literal["one", "two"]
    count: int
    tools: list[McpToolOut]


@router.get("/tools", response_model=McpToolsResponse)
async def list_mcp_tools(config: Literal["one", "two"] = "one") -> McpToolsResponse:
    """A fresh connect-discover-disconnect per call -- connecting is cheap
    now that the servers no longer load their own embedder/reranker copy
    (see policy_search_server.py's module docstring), so there is no
    reconnect-overhead reason to hold a persistent client across requests.
    This IS the live tools/list result, never a stored list.
    """
    client = mcp_agent.build_client(config)
    async with client:
        tools = await mcp_agent.discover_tools(client)
    return McpToolsResponse(config_label=config, count=len(tools), tools=[McpToolOut(**t) for t in tools])


# --- Diff proofs (requirement #2, 30 pts) -----------------------------------


class DiffResponse(BaseModel):
    diff_text: str
    is_empty: bool


@router.get("/agent-diff", response_model=DiffResponse)
async def get_agent_diff() -> DiffResponse:
    diff_text, is_empty = mcp_diff.agent_diff()
    (_AGENTS_DIR / "agent_diff.txt").write_text(diff_text, encoding="utf-8")
    return DiffResponse(diff_text=diff_text, is_empty=is_empty)


@router.get("/config-diff", response_model=DiffResponse)
async def get_config_diff() -> DiffResponse:
    diff_text, is_empty = mcp_diff.config_diff()
    (_AGENTS_DIR / "config_diff.txt").write_text(diff_text, encoding="utf-8")
    return DiffResponse(diff_text=diff_text, is_empty=is_empty)


# --- Wire capture (requirement #4, 25 pts) ----------------------------------


class WireCaptureRequest(BaseModel):
    server: Literal["policy-search", "hris"]
    tool: str
    args: dict = {}


class WireCaptureResponse(BaseModel):
    frames: list[dict]
    annotations_markdown: str


@router.post("/wire-capture", response_model=WireCaptureResponse)
async def wire_capture(request: WireCaptureRequest) -> WireCaptureResponse:
    frames = await mcp_wire_capture.capture(request.server, request.tool, request.args)
    (_AGENTS_DIR / "wire.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
    annotations_markdown = (_AGENTS_DIR / "wire_annotations.md").read_text(encoding="utf-8")
    return WireCaptureResponse(frames=frames, annotations_markdown=annotations_markdown)


# --- Recoverable-error before/after (requirement #5, 20 pts) ---------------


def _load_error_demo_state() -> dict:
    if _ERROR_DEMO_STATE_PATH.exists():
        return json.loads(_ERROR_DEMO_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_error_demo_state(state: dict) -> None:
    _ERROR_DEMO_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _maybe_write_transcript(state: dict) -> None:
    """Writes the required error_before_after.md as soon as both halves have
    been run at least once -- overwritten on every subsequent run of either
    half, always reflecting the latest captured pair.
    """
    if "before" not in state or "after" not in state:
        return
    lines = ["# Error before/after transcript", "", f"**Question:** {state['before']['question']}", ""]
    for label in ("before", "after"):
        entry = state[label]
        lines.append(f"## {label.capitalize()}")
        for i, step in enumerate(entry["steps"], 1):
            lines.append(f"{i}. `{step['action']}({step['action_input']})` -> {step['observation']}")
        lines.append(f"\n**Final answer:** {entry['answer']}\n")
    (_AGENTS_DIR / "error_before_after.md").write_text("\n".join(lines), encoding="utf-8")


class ErrorDemoRequest(BaseModel):
    version: Literal["before", "after"]


class ErrorDemoResponse(BaseModel):
    question: str
    answer: str | None
    steps: list[StepOut]
    terminated_reason: str | None


@router.post("/error-demo", response_model=ErrorDemoResponse)
async def error_demo(request: ErrorDemoRequest, settings: Settings = Depends(get_settings)) -> ErrorDemoResponse:
    result = await mcp_error_demo.run_error_demo(request.version, settings.groq_api_key, settings.groq_generator_model)
    state = _load_error_demo_state()
    state[request.version] = result
    _save_error_demo_state(state)
    _maybe_write_transcript(state)
    return ErrorDemoResponse(**result)


# --- Ask via MCP (ties everything together as one usable demo) -------------


class McpAskRequest(BaseModel):
    question: str


class McpAskResponse(BaseModel):
    question: str
    answer: str | None
    steps: list[StepOut]
    terminated_reason: str | None


@router.post("/ask", response_model=McpAskResponse)
async def ask_via_mcp(request: McpAskRequest, settings: Settings = Depends(get_settings)) -> McpAskResponse:
    client = mcp_agent.build_client("two")
    async with client:
        result = await mcp_agent.run_agent(
            request.question, client, settings.groq_api_key, settings.groq_generator_model, Budgets()
        )
    return McpAskResponse(
        question=request.question,
        answer=result.answer,
        terminated_reason=result.terminated_reason,
        steps=[
            StepOut(thought=s.thought, action=s.action, action_input=s.action_input, observation=s.observation)
            for s in result.steps
        ],
    )


# --- Risk note ---------------------------------------------------------------


class RiskNoteResponse(BaseModel):
    text: str | None


@router.get("/risk-note", response_model=RiskNoteResponse)
async def get_risk_note() -> RiskNoteResponse:
    path = _AGENTS_DIR / "risk_note.md"
    return RiskNoteResponse(text=path.read_text(encoding="utf-8") if path.exists() else None)
