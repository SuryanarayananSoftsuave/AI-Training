from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langfuse import Langfuse

from app.core.config import Settings, get_settings
from app.core.dependencies import get_langfuse_client, get_llm_clients, get_store, get_trace_store
from app.llm.base import LLMClient
from app.models.schemas import ChatRequest
from app.retrieval.qdrant_store import QdrantStore
from app.services.chat_service import answer_query
from app.trace.store import TraceStore

router = APIRouter(tags=["chat"])


async def _sse_events(
    request: ChatRequest,
    settings: Settings,
    store: QdrantStore,
    llm_clients: dict[str, LLMClient],
    trace_store: TraceStore,
    langfuse: Langfuse | None,
) -> AsyncIterator[str]:
    """Frames `answer_query`'s ("delta"|"final", payload) stream as SSE.
    Delta payloads are JSON-wrapped (`{"text": ...}`), not sent as raw text,
    so a chunk containing a literal newline still fits on one `data:` line
    without needing SSE's multi-line-data continuation syntax.
    """
    async for event_type, payload in answer_query(
        query=request.query,
        use_keyword_search=request.use_keyword_search,
        use_query_expansion=request.use_query_expansion,
        use_mmr=request.use_mmr,
        top_k=request.top_k,
        doc_ids=request.doc_ids,
        settings=settings,
        store=store,
        generator=llm_clients[request.generator_provider],
        judge=llm_clients[request.judge_provider],
        generator_provider=request.generator_provider,
        judge_provider=request.judge_provider,
        generator_temperature=request.generator_temperature,
        judge_temperature=request.judge_temperature,
        trace_store=trace_store,
        langfuse=langfuse,
    ):
        data = json.dumps({"text": payload}) if event_type == "delta" else payload.model_dump_json()
        yield f"event: {event_type}\ndata: {data}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
    llm_clients: dict[str, LLMClient] = Depends(get_llm_clients),
    trace_store: TraceStore = Depends(get_trace_store),
    langfuse: Langfuse | None = Depends(get_langfuse_client),
) -> StreamingResponse:
    return StreamingResponse(
        _sse_events(request, settings, store, llm_clients, trace_store, langfuse), media_type="text/event-stream"
    )
