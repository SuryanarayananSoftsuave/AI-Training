"""Week 7 bonus (W7-Task-Set-C.md Sec.5): runs all 3 scripted 30-turn
conversations from agents/fixtures/long_conversations.json end to end, each
with one simulated process restart at a different point (turn 10/15/20, for
variety), and reports what actually happened -- not an assumed outcome.
Writes agents/long_conversation_race.csv (new file, never touches
agents/race.csv) and agents/bonus_conversation_findings.txt (the required
write-up naming the destroyed detail and the question that broke, per
conversation, judged objectively against the fixture's planted_detail
fields rather than by eyeballing the transcript).

A real restart is not simulated by starting/stopping the FastAPI process --
see run_conversation_demo.py's docstring for why that's impractical and why
discarding in-memory state while re-reading only SessionStore is an
accurate stand-in (window/summary never touch disk either way).

Run from backend/, with PYTHONPATH set to it (real ~114 Groq calls across
all 3 conversations -- expect several minutes, run as a background task):

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/race_long_conversations.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore
from groq import AsyncGroq

from agents.conversation import ConversationState, Turn, append_turn
from agents.react_agent import Budgets, run_agent

_FIXTURES_PATH = Path(__file__).parent.parent / "agents" / "fixtures" / "long_conversations.json"
_CSV_PATH = Path(__file__).parent.parent / "agents" / "long_conversation_race.csv"
_FINDINGS_PATH = Path(__file__).parent.parent / "agents" / "bonus_conversation_findings.txt"

# Different restart point per conversation, purely for variety across the 3
# required runs -- the mechanism being demonstrated (jurisdiction survives,
# everything else resets) doesn't depend on which turn it happens at.
_RESTART_AT = {0: 10, 1: 15, 2: 20}


async def _run_conversation(index: int, conversation: dict, session_store: SessionStore, groq_client: AsyncGroq, settings, store: QdrantStore) -> dict:
    session_id = conversation["session_id"]
    detail = conversation["planted_detail"]
    restart_at = _RESTART_AT[index]

    state = ConversationState(session_id=session_id)
    total_tokens = 0
    total_cost = 0.0
    folds = 0
    started = time.monotonic()
    jurisdiction_before_restart: str | None = None
    jurisdiction_after_restart: str | None = None
    recall_answer: str | None = None

    for turn in conversation["turns"]:
        n = turn["n"]
        result = await run_agent(
            turn["question"], settings, store, settings.groq_api_key, settings.groq_generator_model,
            Budgets(max_tokens=8000), conversation=state, session_store=session_store,
        )
        total_tokens += result.total_tokens
        total_cost += result.cost_usd
        summarized = await append_turn(
            state, Turn(question=turn["question"], answer=result.answer), groq_client, settings.groq_generator_model,
            settings.conversation_window_turns, settings.conversation_fold_batch_size,
        )
        if summarized:
            folds += 1
        if n == len(conversation["turns"]):
            recall_answer = result.answer

        if n == restart_at:
            jurisdiction_before_restart = state.jurisdiction
            state = ConversationState(session_id=session_id, jurisdiction=session_store.get_jurisdiction(session_id))
            jurisdiction_after_restart = state.jurisdiction

    manager_last_name = detail["manager"].split()[-1]
    manager_survived = manager_last_name.lower() in (recall_answer or "").lower()
    office_survived = detail["office"].lower() in (recall_answer or "").lower()

    return {
        "session_id": session_id,
        "employee_id": conversation["employee_id"],
        "total_turns": len(conversation["turns"]),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "wall_clock_s": round(time.monotonic() - started, 2),
        "folds": folds,
        "restart_at_turn": restart_at,
        "jurisdiction_before_restart": jurisdiction_before_restart,
        "jurisdiction_after_restart": jurisdiction_after_restart,
        "restart_preserved_jurisdiction": jurisdiction_before_restart == jurisdiction_after_restart and jurisdiction_after_restart is not None,
        "recall_question": conversation["turns"][-1]["question"],
        "recall_answer": recall_answer,
        "planted_manager": detail["manager"],
        "planted_office": detail["office"],
        "manager_survived": manager_survived,
        "office_survived": office_survived,
    }


async def main() -> None:
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    session_store = SessionStore(settings.session_store_path)
    groq_client = AsyncGroq(api_key=settings.groq_api_key, max_retries=6)

    conversations = json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))["conversations"]
    results = []
    for i, conv in enumerate(conversations):
        print(f"--- Conversation {i + 1}/{len(conversations)}: {conv['session_id']} (restart at turn {_RESTART_AT[i]}) ---")
        result = await _run_conversation(i, conv, session_store, groq_client, settings, store)
        print(
            f"  {result['total_turns']} turns, {result['folds']} folds, {result['total_tokens']} tokens, "
            f"${result['total_cost_usd']}, {result['wall_clock_s']}s\n"
            f"  restart at turn {result['restart_at_turn']}: jurisdiction before={result['jurisdiction_before_restart']!r} "
            f"after={result['jurisdiction_after_restart']!r} (preserved={result['restart_preserved_jurisdiction']})\n"
            f"  recall answer: {result['recall_answer']!r}\n"
            f"  manager ({result['planted_manager']}) survived: {result['manager_survived']}  "
            f"office ({result['planted_office']}) survived: {result['office_survived']}\n"
        )
        results.append(result)

    with _CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"long_conversation_race.csv written to {_CSV_PATH}")

    findings_lines = [
        "Week 7 bonus -- 3 real 30-turn conversations run live through the ReAct agent's",
        "sliding-window memory (window=6 turns, fold batch=3), one simulated process",
        "restart per conversation. Full transcripts are reproducible via",
        "scripts/race_long_conversations.py against agents/fixtures/long_conversations.json.",
        "",
    ]
    for r in results:
        findings_lines.append(f"## {r['session_id']} (employee {r['employee_id']})")
        findings_lines.append(
            f"Restart at turn {r['restart_at_turn']}: jurisdiction survived = {r['restart_preserved_jurisdiction']} "
            f"({r['jurisdiction_before_restart']!r} -> {r['jurisdiction_after_restart']!r}). Window/summary were "
            f"discarded and rebuilt empty, as designed -- only jurisdiction was re-hydrated from SessionStore."
        )
        findings_lines.append(f"Planted at turn 1: manager={r['planted_manager']!r}, office={r['planted_office']!r}.")
        findings_lines.append(f"{r['folds']} summarization folds happened before turn 30 re-compressed that detail repeatedly.")
        findings_lines.append(f"Turn 30 recall question: {r['recall_question']!r}")
        findings_lines.append(f"Turn 30 actual answer: {r['recall_answer']!r}")
        if r["manager_survived"] and r["office_survived"]:
            findings_lines.append("RESULT: both details survived every fold -- no destroyed detail observed for this conversation.")
        else:
            lost = []
            if not r["manager_survived"]:
                lost.append(f"the manager's name ({r['planted_manager']})")
            if not r["office_survived"]:
                lost.append(f"the office location ({r['planted_office']})")
            findings_lines.append(
                f"RESULT: summarization destroyed {' and '.join(lost)} -- by turn 30 the agent could not answer "
                f"the recall question ({r['recall_question']!r}) using only the surviving window+summary context."
            )
        findings_lines.append("")

    _FINDINGS_PATH.write_text("\n".join(findings_lines), encoding="utf-8")
    print(f"bonus_conversation_findings.txt written to {_FINDINGS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
