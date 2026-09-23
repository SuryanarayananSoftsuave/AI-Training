"""Week 7: run the ReAct agent on a single question, one command.

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/run_agent_demo.py "What is the notice period for employee E5 if they resign?"
"""
from __future__ import annotations

import asyncio
import sys

from app.core.config import get_settings
from app.retrieval.qdrant_store import QdrantStore
from agents.react_agent import Budgets, run_agent


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the notice period for employee E1 if they resign?"
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    r = await run_agent(question, settings, store, settings.groq_api_key, settings.groq_generator_model, Budgets())

    print(f"Question: {question}\n")
    for i, s in enumerate(r.steps, 1):
        print(f"Step {i}: thought={s.thought!r}")
        print(f"         action={s.action}({s.action_input}) -> {s.observation}")
    print(f"\nAnswer: {r.answer}")
    print(f"Laps: {r.iterations}  Tokens: {r.total_tokens}  Time: {r.wall_clock_s:.2f}s  Terminated: {r.terminated_reason or 'normally'}")


if __name__ == "__main__":
    asyncio.run(main())
