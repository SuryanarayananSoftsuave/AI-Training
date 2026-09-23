"""Week 7 bonus: the one fact (an employee's jurisdiction) that must survive
a full backend process restart. Reuses DocumentRegistry's exact atomic-write
pattern (app/registry/json_store.py) -- FileLock sidecar guarding a
read-modify-write cycle, every write via tempfile + fsync + os.replace so a
crash mid-write can't corrupt it -- rather than inventing a second
persistence mechanism for what is otherwise the same problem (a small
JSON store keyed by an id).

Deliberately scoped to jurisdiction only, per the bonus's own wording --
conversation window/summary are NOT persisted here and are expected to be
lost on restart; that contrast is what the exercise is testing.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


class SessionStore:
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
        fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_sessions_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def list_all(self) -> dict[str, dict]:
        with self._lock:
            return self._read()

    def get_jurisdiction(self, session_id: str) -> str | None:
        with self._lock:
            record = self._read().get(session_id)
        return record["jurisdiction"] if record else None

    def set_jurisdiction(self, session_id: str, employee_id: str, jurisdiction: str) -> None:
        with self._lock:
            data = self._read()
            data[session_id] = {
                "employee_id": employee_id,
                "jurisdiction": jurisdiction,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write(data)
