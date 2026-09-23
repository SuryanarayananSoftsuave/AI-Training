"""Week 7 demo endpoint: runs a single employee question through both the
ReAct agent and the fixed workflow, so the UI can show the live step-by-step
trace side by side with the workflow's answer, for any question the user
picks or types.
"""
from __future__ import annotations

import csv
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.dependencies import get_agent_conversations, get_session_store, get_store
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore

from agents.conversation import ConversationState, Turn, append_turn
from agents.dispatcher import run_dispatch
from agents.react_agent import Budgets, run_agent
from agents.workflow import run_workflow
from groq import AsyncGroq

router = APIRouter(prefix="/agents", tags=["agents"])

# Where scripts/race_agent_vs_workflow.py, race_dispatcher.py, and
# branching_case_demo.txt's live test wrote their real, already-verified
# output. /agents/results below just reads these back for the UI -- it
# never regenerates or fabricates them.
_AGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "agents"


def _read_csv_rows(filename: str) -> list[dict]:
    path = _AGENTS_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_text(filename: str) -> str | None:
    path = _AGENTS_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else None


class AgentAskRequest(BaseModel):
    question: str


class StepOut(BaseModel):
    thought: str
    action: str
    action_input: dict
    observation: str


class AgentAskResponse(BaseModel):
    question: str
    agent_answer: str | None
    agent_steps: list[StepOut]
    agent_iterations: int
    agent_tokens: int
    agent_latency_s: float
    agent_terminated_reason: str | None
    workflow_answer: str | None
    workflow_tokens: int
    workflow_latency_s: float
    workflow_error: str | None


@router.post("/ask", response_model=AgentAskResponse)
async def ask(
    request: AgentAskRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> AgentAskResponse:
    agent_result = await run_agent(
        request.question, settings, store, settings.groq_api_key, settings.groq_generator_model, Budgets(),
    )
    workflow_result = await run_workflow(
        request.question, settings, store, settings.groq_api_key, settings.groq_generator_model,
    )
    return AgentAskResponse(
        question=request.question,
        agent_answer=agent_result.answer,
        agent_steps=[StepOut(thought=s.thought, action=s.action, action_input=s.action_input, observation=s.observation) for s in agent_result.steps],
        agent_iterations=agent_result.iterations,
        agent_tokens=agent_result.total_tokens,
        agent_latency_s=agent_result.wall_clock_s,
        agent_terminated_reason=agent_result.terminated_reason,
        workflow_answer=workflow_result.answer,
        workflow_tokens=workflow_result.total_tokens,
        workflow_latency_s=workflow_result.wall_clock_s,
        workflow_error=workflow_result.error,
    )


class DispatchRequest(BaseModel):
    question: str


class DispatchResponse(BaseModel):
    question: str
    system_used: str
    reason: str
    answer: str | None
    total_tokens: int
    cost_usd: float
    wall_clock_s: float
    fallback_used: bool
    error: str | None
    agent_steps: list[StepOut]
    terminated_reason: str | None


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch(
    request: DispatchRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> DispatchResponse:
    result = await run_dispatch(request.question, settings, store, settings.groq_api_key, settings.groq_generator_model)
    return DispatchResponse(
        question=result.question,
        system_used=result.system_used,
        reason=result.reason,
        answer=result.answer,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
        wall_clock_s=result.wall_clock_s,
        fallback_used=result.fallback_used,
        error=result.error,
        agent_steps=[StepOut(thought=s.thought, action=s.action, action_input=s.action_input, observation=s.observation) for s in result.agent_steps],
        terminated_reason=result.terminated_reason,
    )


class ConversationAskRequest(BaseModel):
    session_id: str
    question: str


class ConversationAskResponse(BaseModel):
    session_id: str
    answer: str | None
    turn_number: int
    window_size: int
    summary: str
    summarized_this_turn: bool
    jurisdiction_known: str | None
    agent_terminated_reason: str | None


@router.post("/chat", response_model=ConversationAskResponse)
async def chat(
    request: ConversationAskRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
    session_store: SessionStore = Depends(get_session_store),
    conversations: dict[str, ConversationState] = Depends(get_agent_conversations),
) -> ConversationAskResponse:
    """Week 7 bonus: multi-turn conversation over the ReAct agent. A newly
    created ConversationState (either genuinely the first call for this
    session_id, or the first call after a real process restart emptied
    `conversations`) is seeded with whatever jurisdiction survived in
    session_store -- the one fact this exercise requires to survive a
    restart, everything else here does not.
    """
    state = conversations.get(request.session_id)
    if state is None:
        state = ConversationState(session_id=request.session_id, jurisdiction=session_store.get_jurisdiction(request.session_id))
        conversations[request.session_id] = state

    result = await run_agent(
        request.question, settings, store, settings.groq_api_key, settings.groq_generator_model,
        Budgets(max_tokens=8000), conversation=state, session_store=session_store,
    )

    client = AsyncGroq(api_key=settings.groq_api_key, max_retries=6)
    summarized = await append_turn(
        state, Turn(question=request.question, answer=result.answer), client, settings.groq_generator_model,
        settings.conversation_window_turns, settings.conversation_fold_batch_size,
    )

    return ConversationAskResponse(
        session_id=request.session_id,
        answer=result.answer,
        turn_number=state.turn_count,
        window_size=len(state.window),
        summary=state.summary,
        summarized_this_turn=summarized,
        jurisdiction_known=state.jurisdiction,
        agent_terminated_reason=result.terminated_reason,
    )


class ResultsResponse(BaseModel):
    race_rows: list[dict]
    dispatch_race_rows: list[dict]
    verdict: str | None
    budget_termination_log: str | None
    branching_case_demo: str | None
    third_tool_diff: str | None
    persisted_sessions: dict[str, dict]


@router.get("/results", response_model=ResultsResponse)
async def results(session_store: SessionStore = Depends(get_session_store)) -> ResultsResponse:
    """Serves the real, already-generated Week 7 deliverables straight off
    disk for the UI's showcase panel -- nothing here is computed live or
    re-run; it's exactly what scripts/race_agent_vs_workflow.py,
    race_dispatcher.py and the live branching-case test already produced.
    """
    return ResultsResponse(
        race_rows=_read_csv_rows("race.csv"),
        dispatch_race_rows=_read_csv_rows("dispatch_race.csv"),
        verdict=_read_text("verdict.txt"),
        budget_termination_log=_read_text("budget_termination_log.txt"),
        branching_case_demo=_read_text("branching_case_demo.txt"),
        third_tool_diff=_read_text("third_tool_diff.md"),
        persisted_sessions=session_store.list_all(),
    )
