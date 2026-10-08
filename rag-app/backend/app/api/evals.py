"""Live Week 6 eval endpoint: re-runs every case fresh against the real
pipeline (SSE progress, same framing as /chat) and returns per-check pass
rates plus a real precision/recall/F1 confusion matrix for the refusal
check and, once evals/labels_25.json exists, for the judge's grounded
verdict against blind human labels. See evals/metrics.py for why those are
two different kinds of metric.
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.dependencies import get_llm_clients, get_store
from app.llm.base import LLMClient
from app.retrieval.qdrant_store import QdrantStore
from evals.metrics import build_report
from evals.week6_eval import (
    LABELING_CRITERION,
    LABELS_PATH,
    RESULTS_PATH,
    grade_one,
    load_cases,
    load_labels_state,
    load_raw_answers,
    record_label_atomic,
)

router = APIRouter(prefix="/evals", tags=["evals"])
logger = logging.getLogger(__name__)


class Week6RunRequest(BaseModel):
    provider: str = "gemini"
    include_regressions: bool = True


def _load_labels() -> dict[str, bool] | None:
    if not LABELS_PATH.exists():
        return None
    state = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return state.get("labels") or None


def _log_report(report: dict) -> None:
    """Writes the metrics report to the app's own logger (console +
    data/logs/success.log, per main.py's _configure_logging) -- a durable,
    greppable trace of a run's accuracy/precision/recall, not just a number
    shown once in the UI and gone on the next page refresh.
    """
    logger.info(
        "Week 6 eval finished: %d case(s), overall pass rate %.1f%%", report["n_cases"], report["overall_pass_rate"],
    )
    for name, v in report["check_accuracy"].items():
        logger.info("  check accuracy: %-28s %d/%d (%.1f%%)", name, v["passed"], v["total"], v["accuracy"])

    rc = report["refusal_confusion"]
    logger.info(
        "  refusal confusion: TP=%d FP=%d FN=%d TN=%d  precision=%s recall=%s f1=%s accuracy=%s",
        rc["tp"], rc["fp"], rc["fn"], rc["tn"], rc["precision"], rc["recall"], rc["f1"], rc["accuracy"],
    )

    jh = report["judge_vs_human"]
    if jh is not None:
        logger.info(
            "  judge-vs-human (n=%d): TP=%d FP=%d FN=%d TN=%d  precision=%s recall=%s f1=%s accuracy=%s",
            jh["n"], jh["tp"], jh["fp"], jh["fn"], jh["tn"], jh["precision"], jh["recall"], jh["f1"], jh["accuracy"],
        )
    else:
        logger.info("  judge-vs-human: not available yet -- run scripts/label_answers.py to write evals/labels_25.json")


async def _sse_events(
    request: Week6RunRequest, settings: Settings, store: QdrantStore, llm_clients: dict[str, LLMClient],
) -> AsyncIterator[str]:
    generator = llm_clients[request.provider]
    cases = load_cases(include_regressions=request.include_regressions)
    total = len(cases)
    results: list[dict] = []

    logger.info("Week 6 eval run started: %d case(s), provider=%s", total, request.provider)

    for i, case in enumerate(cases, 1):
        yield f"event: progress\ndata: {json.dumps({'i': i, 'total': total, 'id': case['id'], 'status': 'running'})}\n\n"
        t0 = time.monotonic()
        try:
            result = await grade_one(case, settings, store, generator, request.provider)
        except Exception as exc:
            # See app/api/agents.py::_trajectory_sse_events for why this
            # frame exists -- without it, a mid-run failure just drops the
            # connection with no indication of which case failed.
            yield f"event: error\ndata: {json.dumps({'i': i, 'total': total, 'id': case['id'], 'message': str(exc)})}\n\n"
            return
        results.append(result)
        logger.info(
            "[%d/%d] %s: %s (judge_verdict=%s via %s, %.1fs)",
            i, total, case["id"], "PASS" if result["passed"] else "FAIL",
            result["judge_verdict"], result["judged_by"], time.monotonic() - t0,
        )
        yield (
            "event: progress\ndata: "
            f"{json.dumps({'i': i, 'total': total, 'id': case['id'], 'status': 'done', 'passed': result['passed'], 'judge_verdict': result['judge_verdict']})}"
            "\n\n"
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = build_report(results, cases, _load_labels())
    _log_report(report)
    yield f"event: final\ndata: {json.dumps({'results': results, 'report': report})}\n\n"


@router.post("/week6/run")
async def run_week6(
    request: Week6RunRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
    llm_clients: dict[str, LLMClient] = Depends(get_llm_clients),
) -> StreamingResponse:
    return StreamingResponse(_sse_events(request, settings, store, llm_clients), media_type="text/event-stream")


class LabelingStateResponse(BaseModel):
    criterion: str
    total: int
    labeled_count: int
    remaining_ids: list[str]
    all_labeled: bool


@router.get("/week6/labeling-state", response_model=LabelingStateResponse)
async def week6_labeling_state() -> LabelingStateResponse:
    """Progress only -- never a verdict -- so the UI panel can render a
    progress bar without any risk of leaking the judge's opinion into a
    label that's supposed to be blind to it.
    """
    answers = load_raw_answers()
    current_ids = {a["id"] for a in answers}
    state = load_labels_state()
    # Intersect with the CURRENT answer set -- labels_25.json can outlive
    # the fixture it was labeled against (cases get added/renamed via
    # scripts/extract_regression_candidates.py), and measure_judge_agreement.py
    # already guards against exactly this staleness; these two endpoints
    # didn't, which could push labeled_count above total.
    labeled = set(state["labels"]) & current_ids
    remaining = [a["id"] for a in answers if a["id"] not in labeled]
    return LabelingStateResponse(
        criterion=state.get("criterion", LABELING_CRITERION),
        total=len(answers),
        labeled_count=len(labeled),
        remaining_ids=remaining,
        all_labeled=not remaining,
    )


class LabelCitationOut(BaseModel):
    marker: int
    filename: str
    page_number: int | None


class NextUnlabeledResponse(BaseModel):
    case_id: str | None
    mode: str | None
    question: str | None
    answer: str | None
    citations: list[LabelCitationOut]


@router.get("/week6/next-unlabeled", response_model=NextUnlabeledResponse)
async def week6_next_unlabeled() -> NextUnlabeledResponse:
    """The next case still needing a blind label. Deliberately excludes
    `judgment`/`judge_verdict` from the response entirely -- not just
    unrendered by the frontend -- so a blind label can never be influenced
    by a verdict the client never even received.
    """
    answers = load_raw_answers()
    state = load_labels_state()
    labeled = set(state["labels"])
    remaining = [a for a in answers if a["id"] not in labeled]
    if not remaining:
        return NextUnlabeledResponse(case_id=None, mode=None, question=None, answer=None, citations=[])
    ans = remaining[0]
    citations = [
        LabelCitationOut(marker=c["marker"], filename=c["filename"], page_number=c.get("page_number"))
        for c in ans["citations"]
    ]
    return NextUnlabeledResponse(case_id=ans["id"], mode=ans["mode"], question=ans["question"], answer=ans["answer"], citations=citations)


class LabelSubmitRequest(BaseModel):
    case_id: str
    grounded: bool


class LabelSubmitResponse(BaseModel):
    labeled_count: int
    total: int
    all_labeled: bool


@router.post("/week6/label", response_model=LabelSubmitResponse)
async def week6_submit_label(request: LabelSubmitRequest) -> LabelSubmitResponse:
    answers = load_raw_answers()
    current_ids = {a["id"] for a in answers}
    try:
        state = record_label_atomic(request.case_id, request.grounded)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    labeled_count = len(set(state["labels"]) & current_ids)
    return LabelSubmitResponse(labeled_count=labeled_count, total=len(answers), all_labeled=labeled_count >= len(answers))
