from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

_EMPTY_STATE: dict = {
    "seed": None,
    "sampled_trace_ids": [],
    "sampled_at": None,
    "sentences": {},
    "modes": [],
    "prediction": None,
    "benchmark_note": None,
}


class Week5AnalysisStore:
    """Durable state behind the Week 5 Trace Explorer UI panel: the seeded
    sample, one open-coding sentence per trace, named failure modes, the
    dated prediction, and the benchmark-blindspot note.

    Same atomic-write convention as `app/registry/json_store.py::DocumentRegistry`
    (temp file + os.replace, sidecar `.lock`) -- this is small,
    infrequently-written state (one save per button click), not a
    high-throughput append log like `TraceStore`, so a whole-file
    read-modify-write is the right fit here too.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self._path) + ".lock")

    def _read(self) -> dict:
        if not self._path.exists():
            return json.loads(json.dumps(_EMPTY_STATE))  # deep copy
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {**_EMPTY_STATE, **data}

    def _write(self, data: dict) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_week5_")
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

    def get(self) -> dict:
        with self._lock:
            return self._read()

    def set_sample(self, seed: int, trace_ids: list[str]) -> dict:
        """A genuinely NEW sample (different seed / first sample) resets
        every downstream analysis field -- modes/sentences/prediction built
        against a PRIOR sample would otherwise survive silently, letting
        render_deliverables() compute a mode's percentage against the new
        sample's size while still counting trace_ids that may not even be
        in it (can exceed 100%, or just misrepresent the distribution).
        """
        with self._lock:
            data = self._read()
            data["seed"] = seed
            data["sampled_trace_ids"] = trace_ids
            data["sampled_at"] = datetime.now(timezone.utc).isoformat()
            data["sentences"] = {}
            data["modes"] = []
            data["prediction"] = None
            data["benchmark_note"] = None
            self._write(data)
            return data

    def set_sentence(self, trace_id: str, sentence: str) -> dict:
        with self._lock:
            data = self._read()
            data["sentences"][trace_id] = sentence
            self._write(data)
            return data

    def set_modes(self, modes: list[dict]) -> dict:
        with self._lock:
            data = self._read()
            data["modes"] = modes
            self._write(data)
            return data

    def set_prediction(self, mode: str, change: str, expected_delta: str) -> dict:
        with self._lock:
            data = self._read()
            data["prediction"] = {
                "mode": mode,
                "change": change,
                "expected_delta": expected_delta,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write(data)
            return data

    def set_benchmark_note(self, text: str) -> dict:
        with self._lock:
            data = self._read()
            data["benchmark_note"] = text
            self._write(data)
            return data


def render_deliverables(state: dict, questions_by_id: dict[str, str]) -> dict[str, str]:
    """Formats the stored state into the exact rubric-shaped files
    (W5-Task-Set-C.md's expected output: notes.md, taxonomy.md,
    prediction.txt). Pure function -- callers decide whether/where to write
    these to disk (see app/api/analysis.py's export route). The
    benchmark-blindspot note is folded into taxonomy.md's bottom rather than
    a 4th file, since the rubric bundles it with "notes.md ... and the
    3-sentence benchmark note" as one write-up, not a separate deliverable.
    """
    lines_notes = [
        "# Week 5 Open-Coding Notes",
        "",
        f"Seed: `{state['seed']}`  ·  Sampled at: {state['sampled_at']}",
        f"Sample size: {len(state['sampled_trace_ids'])}",
        "",
        "One honest observation sentence per trace, written before any clustering into modes.",
        "",
    ]
    for trace_id in state["sampled_trace_ids"]:
        question = questions_by_id.get(trace_id, "(question unavailable)")
        sentence = state["sentences"].get(trace_id, "_(not yet coded)_")
        lines_notes.append(f"- **`{trace_id}`** — *{question}*\n  {sentence}")
    notes_md = "\n".join(lines_notes) + "\n"

    n = len(state["sampled_trace_ids"]) or 1
    lines_tax = [
        "# Week 5 Failure Taxonomy",
        "",
        "| Mode | Count | % | Severity | Example trace_id |",
        "|---|---|---|---|---|",
    ]
    for mode in state["modes"]:
        count = len(mode.get("trace_ids", []))
        pct = round(100 * count / n, 1)
        example = mode["trace_ids"][0] if mode.get("trace_ids") else "-"
        lines_tax.append(f"| {mode['name']} | {count} | {pct}% | {mode.get('severity', '')} | `{example}` |")
    if state.get("benchmark_note"):
        lines_tax += ["", "## Why a public benchmark would have missed this", "", state["benchmark_note"]]
    taxonomy_md = "\n".join(lines_tax) + "\n"

    prediction = state.get("prediction")
    if prediction:
        prediction_txt = (
            f"Mode: {prediction['mode']}\n"
            f"Change: {prediction['change']}\n"
            f"Expected delta: {prediction['expected_delta']}\n"
            f"Written at: {prediction['written_at']}\n"
        )
    else:
        prediction_txt = "(not yet written)\n"

    return {"notes.md": notes_md, "taxonomy.md": taxonomy_md, "prediction.txt": prediction_txt}
