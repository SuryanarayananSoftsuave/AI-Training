from __future__ import annotations

from fastapi import Request
from langfuse import Langfuse

from app.llm.base import LLMClient
from app.registry.json_store import DocumentRegistry
from app.retrieval.qdrant_store import QdrantStore
from app.trace.store import TraceStore


def get_registry(request: Request) -> DocumentRegistry:
    return request.app.state.registry


def get_store(request: Request) -> QdrantStore:
    return request.app.state.store


def get_llm_clients(request: Request) -> dict[str, LLMClient]:
    return request.app.state.llm_clients


def get_trace_store(request: Request) -> TraceStore:
    return request.app.state.trace_store


def get_langfuse_client(request: Request) -> Langfuse | None:
    """None when LANGFUSE_ENABLED is unset/false -- callers must treat a
    disabled Langfuse client as a no-op, not an error.
    """
    return request.app.state.langfuse_client
