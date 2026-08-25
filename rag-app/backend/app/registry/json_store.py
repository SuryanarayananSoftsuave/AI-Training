from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from app.models.schemas import DocumentRecord


class DocumentRegistry:
    """Hash-keyed JSON store tracking every ingested document.

    Guarded by a sidecar `.lock` file so concurrent uploads never race on a
    read-modify-write cycle, and every write lands via a temp-file +
    os.replace() swap so a crash mid-write can never leave readers with a
    truncated or corrupt registry.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self._path) + ".lock")
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, dict]) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_registry_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._lock:
            raw = self._read().get(file_hash)
        return DocumentRecord.model_validate(raw) if raw else None

    def get_by_id(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            data = self._read()
        for raw in data.values():
            if raw["doc_id"] == doc_id:
                return DocumentRecord.model_validate(raw)
        return None

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            data = self._read()
        return [DocumentRecord.model_validate(raw) for raw in data.values()]

    def upsert(self, record: DocumentRecord) -> None:
        with self._lock:
            data = self._read()
            data[record.file_hash] = record.model_dump(mode="json")
            self._write(data)

    def delete_by_id(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            data = self._read()
            target_hash = next((h for h, raw in data.items() if raw["doc_id"] == doc_id), None)
            if target_hash is None:
                return None
            removed = data.pop(target_hash)
            self._write(data)
        return DocumentRecord.model_validate(removed)
