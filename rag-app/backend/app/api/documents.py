from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.dependencies import get_registry, get_store
from app.models.schemas import DocumentListResponse, DocumentRecord, DocumentStatus, UploadResponse
from app.registry.json_store import DocumentRegistry
from app.retrieval.qdrant_store import QdrantStore
from app.services.ingestion_service import run_ingestion

router = APIRouter(prefix="/documents", tags=["documents"])

_UPLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    registry: DocumentRegistry = Depends(get_registry),
    store: QdrantStore = Depends(get_store),
) -> UploadResponse:
    is_pdf_name = (file.filename or "").lower().endswith(".pdf")
    if file.content_type not in ("application/pdf", "application/octet-stream") and not is_pdf_name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="only PDF files are accepted")

    uploads_dir = Path(settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Stream to disk while hashing incrementally — one pass, no second read
    # of the file just to compute its SHA-256.
    digest = hashlib.sha256()
    fd, tmp_path_str = tempfile.mkstemp(dir=uploads_dir, prefix=".upload_")
    tmp_path = Path(tmp_path_str)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
    finally:
        await file.close()

    file_hash = digest.hexdigest()
    existing = registry.get_by_hash(file_hash)

    if existing is not None and existing.status in (DocumentStatus.INDEXED, DocumentStatus.PROCESSING):
        # Already indexed -> a true duplicate, nothing to do. Still
        # (legitimately) processing -> don't fire a second concurrent
        # ingestion of the same content; either it finishes on its own, or
        # a server restart will mark it failed (see startup reconciliation
        # in main.py) so a later re-upload can retry it.
        tmp_path.unlink(missing_ok=True)
        verb = "already indexed as" if existing.status == DocumentStatus.INDEXED else "already being ingested as"
        return UploadResponse(
            doc_id=existing.doc_id,
            status=existing.status,
            duplicate_of=existing.doc_id,
            message=f"identical content {verb} '{existing.original_filename}'",
        )

    if existing is not None:
        # A previous attempt for this exact content failed (or never got
        # past `pending`) -- retry it in place rather than returning a
        # "duplicate" that can never be re-ingested.
        doc_id = existing.doc_id
        stored_path = Path(existing.stored_path)
        tmp_path.replace(stored_path)
    else:
        doc_id = str(uuid4())
        stored_path = uploads_dir / f"{doc_id}.pdf"
        tmp_path.rename(stored_path)

    record = DocumentRecord(
        doc_id=doc_id,
        original_filename=file.filename or f"{doc_id}.pdf",
        stored_path=str(stored_path),
        file_hash=file_hash,
        file_size_bytes=size,
        upload_timestamp=datetime.now(timezone.utc),
        status=DocumentStatus.PROCESSING,
    )
    registry.upsert(record)

    background_tasks.add_task(
        run_ingestion,
        doc_id=doc_id,
        file_hash=file_hash,
        stored_path=str(stored_path),
        original_filename=record.original_filename,
        settings=settings,
        registry=registry,
        store=store,
    )

    return UploadResponse(doc_id=doc_id, status=DocumentStatus.PROCESSING, message="upload accepted, ingestion started")


@router.get("", response_model=DocumentListResponse)
def list_documents(registry: DocumentRegistry = Depends(get_registry)) -> DocumentListResponse:
    docs = sorted(registry.list_all(), key=lambda d: d.upload_timestamp, reverse=True)
    return DocumentListResponse(documents=docs)


@router.get("/{doc_id}", response_model=DocumentRecord)
def get_document(doc_id: str, registry: DocumentRegistry = Depends(get_registry)) -> DocumentRecord:
    record = registry.get_by_id(doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no document with id '{doc_id}'")
    return record


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: str,
    registry: DocumentRegistry = Depends(get_registry),
    store: QdrantStore = Depends(get_store),
) -> None:
    record = registry.delete_by_id(doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no document with id '{doc_id}'")
    store.delete_by_doc_id(doc_id)
    Path(record.stored_path).unlink(missing_ok=True)
