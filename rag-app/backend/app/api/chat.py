from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.dependencies import get_llm_clients, get_store
from app.llm.base import LLMClient
from app.models.schemas import ChatRequest, ChatResponse
from app.retrieval.qdrant_store import QdrantStore
from app.services.chat_service import answer_query

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
    llm_clients: dict[str, LLMClient] = Depends(get_llm_clients),
) -> ChatResponse:
    return answer_query(
        query=request.query,
        use_keyword_search=request.use_keyword_search,
        use_query_expansion=request.use_query_expansion,
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
    )
