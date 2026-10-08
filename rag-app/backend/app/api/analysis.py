"""Week 5 showcase: turns the CLI/hand-edited-file workflow
(scripts/sample_traces.py + manually writing notes.md/taxonomy.md/
prediction.txt) into a guided UI flow. This layer only orchestrates
existing pieces -- TraceStore's real seeded sampling and trace lookup
(unchanged, reused as-is), plus the new Week5AnalysisStore for the
open-coding/taxonomy/prediction state -- it never fabricates trace content
or open-codes a trace on the user's behalf.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.analysis.week5_store import Week5AnalysisStore, render_deliverables
from app.core.dependencies import get_trace_store
from app.trace.store import TraceStore

router = APIRouter(prefix="/analysis/week5", tags=["analysis"])

_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "analysis" / "week5_state.json"
# Mirrors agents/'s role as "this week's raw graded deliverables" -- a
# dedicated folder for the files a mentor/grader would actually open.
_DELIVERABLES_DIR = Path(__file__).resolve().parent.parent.parent / "analysis"
_store = Week5AnalysisStore(_STATE_PATH)


class SampledTraceOut(BaseModel):
    trace_id: str
    timestamp: str | None
    question: str


class SampleResponse(BaseModel):
    seed: int
    sampled_at: str | None
    traces: list[SampledTraceOut]


@router.get("/sample", response_model=SampleResponse)
async def get_sample(seed: int = 42, n: int = 20, trace_store: TraceStore = Depends(get_trace_store)) -> SampleResponse:
    """Returns the stored sample if one already exists for this seed (so
    reopening the panel doesn't silently resample and orphan sentences
    already written against the old sample); only draws a fresh seeded
    sample -- via TraceStore.sample_random, the exact function
    scripts/sample_traces.py already uses -- when none is stored yet or a
    different seed is explicitly requested.
    """
    state = _store.get()
    if not state["sampled_trace_ids"] or state["seed"] != seed:
        pool = trace_store.read_all()
        if not pool:
            raise HTTPException(
                status_code=400, detail="No traces recorded yet -- use the chat or the batch trace generator first."
            )
        sample = trace_store.sample_random(n, seed)
        state = _store.set_sample(seed, [t.trace_id for t in sample])

    all_traces = {t.trace_id: t for t in trace_store.read_all()}
    traces_out = [
        SampledTraceOut(
            trace_id=tid,
            timestamp=all_traces[tid].timestamp.isoformat() if tid in all_traces else None,
            question=all_traces[tid].question if tid in all_traces else "(trace no longer on disk)",
        )
        for tid in state["sampled_trace_ids"]
    ]
    return SampleResponse(seed=state["seed"], sampled_at=state["sampled_at"], traces=traces_out)


class TraceDetailResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    ranked_candidates: list[dict]
    judgment: dict | None
    error: str | None
    timings_ms: dict[str, float]
    sentence: str | None


@router.get("/trace/{trace_id}", response_model=TraceDetailResponse)
async def get_trace_detail(trace_id: str, trace_store: TraceStore = Depends(get_trace_store)) -> TraceDetailResponse:
    trace = trace_store.get_by_id(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
    state = _store.get()
    return TraceDetailResponse(
        trace_id=trace.trace_id,
        question=trace.question,
        answer=trace.answer,
        ranked_candidates=[c.model_dump(mode="json") for c in trace.ranked_candidates],
        judgment=trace.judgment.model_dump(mode="json") if trace.judgment else None,
        error=trace.error,
        timings_ms=trace.timings_ms,
        sentence=state["sentences"].get(trace_id),
    )


class SentenceRequest(BaseModel):
    trace_id: str
    sentence: str


class SentenceResponse(BaseModel):
    coded_count: int
    total: int


@router.post("/sentence", response_model=SentenceResponse)
async def set_sentence(request: SentenceRequest) -> SentenceResponse:
    state = _store.set_sentence(request.trace_id, request.sentence)
    coded = sum(1 for tid in state["sampled_trace_ids"] if tid in state["sentences"])
    return SentenceResponse(coded_count=coded, total=len(state["sampled_trace_ids"]))


class ModeIn(BaseModel):
    name: str
    trace_ids: list[str]
    severity: str
    note: str = ""


class ModesRequest(BaseModel):
    modes: list[ModeIn]


@router.post("/modes")
async def set_modes(request: ModesRequest) -> dict:
    _store.set_modes([m.model_dump() for m in request.modes])
    return {"count": len(request.modes)}


class PredictionRequest(BaseModel):
    mode: str
    change: str
    expected_delta: str


@router.post("/prediction")
async def set_prediction(request: PredictionRequest) -> dict:
    state = _store.set_prediction(request.mode, request.change, request.expected_delta)
    return state["prediction"]


class BenchmarkNoteRequest(BaseModel):
    text: str


@router.post("/benchmark-note")
async def set_benchmark_note(request: BenchmarkNoteRequest) -> dict:
    _store.set_benchmark_note(request.text)
    return {"saved": True}


class ExportResponse(BaseModel):
    files: dict[str, str]
    written_to: str


@router.get("/export", response_model=ExportResponse)
async def export(trace_store: TraceStore = Depends(get_trace_store)) -> ExportResponse:
    """Renders the current state into notes.md/taxonomy.md/prediction.txt
    and writes them to backend/analysis/ -- display + write only, never a
    git add/commit (that stays a manual, reviewed step per the user's own
    instruction).
    """
    state = _store.get()
    all_traces = {t.trace_id: t for t in trace_store.read_all()}
    questions_by_id = {tid: all_traces[tid].question for tid in state["sampled_trace_ids"] if tid in all_traces}
    files = render_deliverables(state, questions_by_id)

    _DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (_DELIVERABLES_DIR / filename).write_text(content, encoding="utf-8")

    return ExportResponse(files=files, written_to=str(_DELIVERABLES_DIR))
