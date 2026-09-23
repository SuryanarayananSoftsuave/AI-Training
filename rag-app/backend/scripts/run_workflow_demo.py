"""Week 7: run the fixed workflow on a single question, one command.

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/run_workflow_demo.py "What is the notice period for employee E5 if they resign?"
"""
from __future__ import annotations

import asyncio
import sys

from app.core.config import get_settings
from app.retrieval.qdrant_store import QdrantStore
from agents.workflow import run_workflow


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the notice period for employee E1 if they resign?"
    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    r = await run_workflow(question, settings, store, settings.groq_api_key, settings.groq_generator_model)

    print(f"Question: {question}\n")
    print(f"Answer: {r.answer}")
    print(f"Tokens: {r.total_tokens}  Time: {r.wall_clock_s:.2f}s  Error: {r.error or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
