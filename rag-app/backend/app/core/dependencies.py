from __future__ import annotations

from fastapi import Request

from app.llm.base import LLMClient
from app.registry.json_store import DocumentRegistry
from app.retrieval.qdrant_store import QdrantStore


def get_registry(request: Request) -> DocumentRegistry:
    return request.app.state.registry


def get_store(request: Request) -> QdrantStore:
    return request.app.state.store


def get_llm_clients(request: Request) -> dict[str, LLMClient]:
    return request.app.state.llm_clients
