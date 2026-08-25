from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client import models

from app.core.config import Settings
from app.ingestion.chunker import chunk_page
from app.ingestion.embedder import embed_passages
from app.ingestion.parser import parse_pdf
from app.ingestion.sparse import embed_sparse
from app.models.schemas import DocumentStatus
from app.registry.json_store import DocumentRegistry
from app.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Embedding is CPU-bound (dense + sparse, per chunk). Letting every upload's
# BackgroundTask race in parallel makes them all thrash the same cores at
# once and finish slower in aggregate than running one at a time -- this
# lock serializes the heavy part of ingestion so uploads process in order,
# with each one finishing at a predictable pace instead of all crawling
# forward together.
_ingestion_lock = threading.Lock()


def run_ingestion(
    doc_id: str,
    file_hash: str,
    stored_path: str,
    original_filename: str,
    settings: Settings,
    registry: DocumentRegistry,
    store: QdrantStore,
) -> None:
    """Parse -> chunk -> embed (dense + sparse) -> upsert -> mark the
    registry entry `indexed` or `failed`. Runs as a FastAPI BackgroundTask,
    after the upload response has already been sent.
    """
    if _ingestion_lock.locked():
        logger.info("ingestion queued for %s (waiting on another document currently ingesting)", original_filename)

    with _ingestion_lock:
        _run_ingestion_locked(doc_id, file_hash, stored_path, original_filename, settings, registry, store)


def _run_ingestion_locked(
    doc_id: str,
    file_hash: str,
    stored_path: str,
    original_filename: str,
    settings: Settings,
    registry: DocumentRegistry,
    store: QdrantStore,
) -> None:
    started = time.monotonic()
    logger.info("ingestion started: %s (doc_id=%s)", original_filename, doc_id)

    try:
        step_start = time.monotonic()
        pages = parse_pdf(stored_path)
        logger.info("parsed %s: %d pages (%.1fs)", original_filename, len(pages), time.monotonic() - step_start)

        step_start = time.monotonic()
        all_chunks = []
        for page in pages:
            all_chunks.extend(
                chunk_page(page.markdown, page.page_number, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
            )
        if not all_chunks:
            raise ValueError("no extractable text found in this PDF")
        num_tables = sum(1 for c in all_chunks if c.content_type == "table")
        logger.info(
            "chunked %s: %d chunks (%d table, %d text) (%.1fs)",
            original_filename, len(all_chunks), num_tables, len(all_chunks) - num_tables, time.monotonic() - step_start,
        )

        texts = [chunk.text for chunk in all_chunks]

        step_start = time.monotonic()
        dense_vectors = embed_passages(settings.embedding_model_name, texts)
        logger.info("embedded %s: %d dense vectors (%.1fs)", original_filename, len(dense_vectors), time.monotonic() - step_start)

        step_start = time.monotonic()
        sparse_vectors = embed_sparse(settings.sparse_model_name, texts)
        logger.info("embedded %s: %d sparse vectors (%.1fs)", original_filename, len(sparse_vectors), time.monotonic() - step_start)

        upload_date = datetime.now(timezone.utc).isoformat()
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i]},
                payload={
                    "doc_id": doc_id,
                    "filename": original_filename,
                    "file_hash": file_hash,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading,
                    "content_type": chunk.content_type,
                    "chunk_index": i,
                    "upload_date": upload_date,
                    "text": chunk.text,
                },
            )
            for i, chunk in enumerate(all_chunks)
        ]

        step_start = time.monotonic()
        store.upsert_chunks(points)
        logger.info("upserted %s: %d points to Qdrant (%.1fs)", original_filename, len(points), time.monotonic() - step_start)

        record = registry.get_by_hash(file_hash)
        if record is None:
            raise RuntimeError("registry entry disappeared mid-ingestion")
        record.status = DocumentStatus.INDEXED
        record.num_pages = len(pages)
        record.num_chunks = len(all_chunks)
        record.error = None
        registry.upsert(record)
        logger.info(
            "ingestion complete: %s (%d pages, %d chunks, %.1fs total)",
            original_filename, len(pages), len(all_chunks), time.monotonic() - started,
        )

    except Exception as exc:
        logger.exception("ingestion failed for %s after %.1fs", original_filename, time.monotonic() - started)
        record = registry.get_by_hash(file_hash)
        if record is not None:
            record.status = DocumentStatus.FAILED
            record.error = str(exc)
            registry.upsert(record)
