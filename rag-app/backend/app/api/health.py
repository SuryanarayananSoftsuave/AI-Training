from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import get_store
from app.models.schemas import HealthResponse
from app.retrieval.qdrant_store import QdrantStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(store: QdrantStore = Depends(get_store)) -> HealthResponse:
    connected = await store.is_connected()
    exists = await store.collection_exists() if connected else False
    return HealthResponse(status="ok" if connected else "degraded", qdrant_connected=connected, collection_exists=exists)
