from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import Langfuse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import agents, chat, documents, evals, health
from app.core.config import get_settings
from app.ingestion.embedder import get_embedder
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.models.schemas import DocumentStatus, ErrorResponse
from app.registry.json_store import DocumentRegistry
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.reranker import get_reranker
from app.trace.store import TraceStore
from agents.conversation import ConversationState


class _BelowLevelFilter(logging.Filter):
    """Lets a record through only if it's strictly below `ceiling` --
    `Handler.setLevel` alone only enforces a lower bound, so the
    success-log handler (which must exclude WARNING/ERROR, not just
    include everything from DEBUG up) needs this on top of it.
    """

    def __init__(self, ceiling: int) -> None:
        super().__init__()
        self._ceiling = ceiling

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self._ceiling


def _configure_logging(success_log_path: str, failure_log_path: str) -> None:
    """Three handlers on the root logger: console (everything, as before --
    unchanged behavior for anyone watching the terminal), success.log
    (DEBUG/INFO only -- normal pipeline progress), failure.log (WARNING and
    above -- everything that actually went wrong). The level split matches
    how this codebase already uses `logger.info` for progress and
    `logger.warning`/`logger.exception` for problems, so no call site
    changes -- only where each line ends up.
    """
    Path(success_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(failure_log_path).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler()

    success_handler = logging.FileHandler(success_log_path, encoding="utf-8")
    success_handler.addFilter(_BelowLevelFilter(logging.WARNING))

    failure_handler = logging.FileHandler(failure_log_path, encoding="utf-8")
    failure_handler.setLevel(logging.WARNING)

    for handler in (console_handler, success_handler, failure_handler):
        handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[console_handler, success_handler, failure_handler])


_configure_logging(get_settings().success_log_path, get_settings().failure_log_path)
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
    await store.ensure_collection()

    registry = DocumentRegistry(settings.registry_path)
    _reconcile_interrupted_ingestions(registry)

    session_store = SessionStore(settings.session_store_path)

    langfuse_client: Langfuse | None = None
    if settings.langfuse_enabled:
        langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse observability enabled (host=%s)", settings.langfuse_host)
    else:
        logger.info("Langfuse observability disabled (LANGFUSE_ENABLED=false)")

    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store
    app.state.session_store = session_store
    # In-memory only, by design: a real restart empties this dict, leaving
    # session_store above as the sole thing that survives it (Week 7 bonus).
    app.state.agent_conversations: dict[str, ConversationState] = {}
    app.state.trace_store = TraceStore(settings.trace_success_log_path, settings.trace_failure_log_path)
    app.state.langfuse_client = langfuse_client
    app.state.llm_clients = {
        "gemini": GeminiClient(settings.gemini_api_key, settings.gemini_generator_model, settings.gemini_judge_model),
        "groq": GroqClient(settings.groq_api_key, settings.groq_generator_model, settings.groq_judge_model),
    }

    logger.info("startup complete")
    yield

    if langfuse_client is not None:
        langfuse_client.flush()


app = FastAPI(title="HR Policy RAG API", version="1.0.0", lifespan=lifespan)

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
app.include_router(agents.router)
app.include_router(evals.router)
