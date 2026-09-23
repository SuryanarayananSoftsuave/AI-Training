"""Week 7 bonus: drives a multi-turn conversation through the ReAct agent's
sliding-window memory (agents/conversation.py) directly -- no server needed,
since starting/stopping the real FastAPI process mid-script isn't
practical. At --restart-at N, the in-memory ConversationState is discarded
and rebuilt from scratch, re-hydrating ONLY jurisdiction from SessionStore
-- an accurate stand-in for a real restart, since window/summary never
touch disk in this design either way (only jurisdiction does, via
app/registry/session_store.py, exactly as a real restart would leave it).

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/run_conversation_demo.py --conversation 1 --restart-at 15
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore
from groq import AsyncGroq

from agents.conversation import ConversationState, Turn, append_turn
from agents.react_agent import Budgets, run_agent

_FIXTURES_PATH = Path(__file__).parent.parent / "agents" / "fixtures" / "long_conversations.json"


def _load_conversation(index: int) -> dict:
    conversations = json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))["conversations"]
    return conversations[index - 1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation", type=int, required=True, help="1-based index into long_conversations.json")
    parser.add_argument("--restart-at", type=int, default=None, help="Turn number (1-based) after which to simulate a process restart")
    args = parser.parse_args()

    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    session_store = SessionStore(settings.session_store_path)
    groq_client = AsyncGroq(api_key=settings.groq_api_key, max_retries=6)

    conversation = _load_conversation(args.conversation)
    session_id = conversation["session_id"]
    state = ConversationState(session_id=session_id)

    for i, turn in enumerate(conversation["turns"], start=1):
        result = await run_agent(
            turn["question"], settings, store, settings.groq_api_key, settings.groq_generator_model,
            Budgets(max_tokens=8000), conversation=state, session_store=session_store,
        )
        summarized = await append_turn(
            state, Turn(question=turn["question"], answer=result.answer), groq_client, settings.groq_generator_model,
            settings.conversation_window_turns, settings.conversation_fold_batch_size,
        )
        print(f"Turn {i}: Q: {turn['question']!r}")
        print(f"         A: {result.answer!r}")
        print(f"         window={len(state.window)} summary_updated={summarized} jurisdiction={state.jurisdiction!r}")
        if summarized:
            print(f"         >>> FOLDED, new summary: {state.summary!r}")

        if args.restart_at == i:
            print(f"\n=== Simulating a full process restart after turn {i} ===")
            print(f"    Before restart: window={len(state.window)} turns, summary={state.summary!r}, jurisdiction={state.jurisdiction!r}")
            state = ConversationState(session_id=session_id, jurisdiction=session_store.get_jurisdiction(session_id))
            print(f"    After restart:  window={len(state.window)} turns, summary={state.summary!r}, jurisdiction={state.jurisdiction!r}\n")

    print(f"\nFinal state: {state.turn_count} turns processed, window={len(state.window)}, jurisdiction={state.jurisdiction!r}")


if __name__ == "__main__":
    asyncio.run(main())
