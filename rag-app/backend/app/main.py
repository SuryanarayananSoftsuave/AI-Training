from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import chat, documents, health
from app.core.config import get_settings
from app.ingestion.embedder import get_embedder
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.models.schemas import DocumentStatus, ErrorResponse
from app.registry.json_store import DocumentRegistry
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.reranker import get_reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _reconcile_interrupted_ingestions(registry: DocumentRegistry) -> None:
    """A `processing` entry cannot legitimately survive a process restart --
    its BackgroundTask died with the old process. Mark any such leftovers
    `failed` so they're no longer silently stuck and can be retried by
    re-uploading the same file.
    """
    for record in registry.list_all():
        if record.status == DocumentStatus.PROCESSING:
            record.status = DocumentStatus.FAILED
            record.error = "ingestion interrupted by a server restart; re-upload to retry"
            registry.upsert(record)
            logger.warning("reset interrupted ingestion on startup: %s (doc_id=%s)", record.original_filename, record.doc_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info("loading embedding model %s ...", settings.embedding_model_name)
    get_embedder(settings.embedding_model_name)
    logger.info("loading reranker model %s ...", settings.reranker_model_name)
    get_reranker(settings.reranker_model_name)

    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    store.ensure_collection()

    registry = DocumentRegistry(settings.registry_path)
    _reconcile_interrupted_ingestions(registry)

    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store
    app.state.llm_clients = {
        "gemini": GeminiClient(settings.gemini_api_key, settings.gemini_generator_model, settings.gemini_judge_model),
        "groq": GroqClient(settings.groq_api_key, settings.groq_generator_model, settings.groq_judge_model),
    }

    logger.info("startup complete")
    yield


app = FastAPI(title="Customer Support RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=str(exc.detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=str(exc.errors())).model_dump(),
    )


app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(health.router)
