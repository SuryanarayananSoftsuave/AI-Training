from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from filelock import FileLock

from app.trace.schema import ChatTrace

logger = logging.getLogger(__name__)


class TraceStore:
    """Append-only JSON-Lines log of chat traces, split into two files by
    outcome: `success_path` (trace.error is None) and `failure_path`
    (trace.error is set). Splitting at write time -- rather than one file
    filtered by the `error` field on read -- means "what's actually
    broken" is a plain file read, not a scan, and keeps the split
    consistent with the app-level success/failure log files configured in
    `main.py`.

    Guarded by the same sidecar `.lock` (via `filelock`) convention as
    `DocumentRegistry` (`app/registry/json_store.py`), one lock per file so
    a success append never blocks on a failure append or vice versa. Unlike
    the registry's whole-file read-modify-write, an append needs no prior
    read -- each write is one `open(..., "a")` + one JSON line under the
    relevant lock.
    """

    def __init__(self, success_path: str | Path, failure_path: str | Path) -> None:
        self._success_path = Path(success_path)
        self._failure_path = Path(failure_path)
        self._success_path.parent.mkdir(parents=True, exist_ok=True)
        self._failure_path.parent.mkdir(parents=True, exist_ok=True)
        self._success_lock = FileLock(str(self._success_path) + ".lock")
        self._failure_lock = FileLock(str(self._failure_path) + ".lock")

    def _path_for(self, trace: ChatTrace) -> tuple[Path, FileLock]:
        if trace.error is None:
            return self._success_path, self._success_lock
        return self._failure_path, self._failure_lock

    def append(self, trace: ChatTrace) -> None:
        path, lock = self._path_for(trace)
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(trace.model_dump_json())
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def _read_file(path: Path, lock: FileLock) -> list[ChatTrace]:
        if not path.exists():
            return []
        with lock:
            raw_lines = path.read_text(encoding="utf-8").splitlines()

        traces = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(ChatTrace.model_validate_json(line))
            except ValueError:
                logger.warning("skipping unparseable trace line in %s (likely a truncated crash-mid-append)", path)
        return traces

    def read_successes(self) -> list[ChatTrace]:
        return self._read_file(self._success_path, self._success_lock)

    def read_failures(self) -> list[ChatTrace]:
        return self._read_file(self._failure_path, self._failure_lock)

    def read_all(self) -> list[ChatTrace]:
        """Both files combined, sorted by timestamp -- the representative
        pool Week 5's seeded sampling draws from (sampling only successes,
        or only failures, would no longer be a representative sample of
        real traffic).
        """
        combined = self.read_successes() + self.read_failures()
        return sorted(combined, key=lambda t: t.timestamp)

    def get_by_id(self, trace_id: str) -> ChatTrace | None:
        for trace in self.read_all():
            if trace.trace_id == trace_id:
                return trace
        return None

    def sample_random(self, n: int, seed: int) -> list[ChatTrace]:
        """Seeded random sample over the combined pool -- calling this
        twice with the same seed against the same trace files must return
        the identical set of trace_ids, which is the provable-not-cherry-
        picked evidence Week 5 requires.
        """
        all_traces = self.read_all()
        rng = random.Random(seed)
        return rng.sample(all_traces, min(n, len(all_traces)))
