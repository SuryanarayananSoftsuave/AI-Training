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

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.dependencies import get_llm_clients, get_store
from app.llm.base import LLMClient
from app.retrieval.qdrant_store import QdrantStore
from evals.metrics import build_report
from evals.week6_eval import LABELS_PATH, RESULTS_PATH, grade_one, load_cases

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
        result = await grade_one(case, settings, store, generator, request.provider)
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
