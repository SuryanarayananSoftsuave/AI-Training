"""Shared logic for the Week 6 harness, used by the 4 scripts in
backend/scripts/: generate_week6_answers.py, run_week6_eval.py,
label_answers.py, measure_judge_agreement.py.

Deliberately mirrors chat_service.answer_query's real retrieve -> rerank ->
generate path (via the exact same function, `skip_judge=True`) rather than
reimplementing it, so a Week 6 case is graded against the same pipeline a
real user's question would hit -- not a simplified stand-in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.llm.base import LLMClient
from app.models.schemas import ChatStreamFinal
from app.retrieval.qdrant_store import QdrantStore
from app.services.chat_service import answer_query
from evals.assertions import ASSERTIONS

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CASES_PATH = _FIXTURES_DIR / "week6_eval_cases.json"
_RAW_ANSWERS_PATH = Path(__file__).parent / "week6_raw_answers.json"
RESULTS_PATH = Path(__file__).parent / "week6_results.json"


def load_cases(include_regressions: bool = True) -> list[dict]:
    data = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    unfilled = [c for c in cases if c["question"].startswith("REGRESSION_CASE_PLACEHOLDER")]
    if unfilled:
        ids = [c["id"] for c in unfilled]
        print(
            f"NOTE: {len(unfilled)} regression case(s) not yet filled in ({ids}) -- run "
            "scripts/extract_regression_candidates.py and edit evals/fixtures/week6_eval_cases.json. "
            "Skipping them for now.\n"
        )
    cases = [c for c in cases if not c["question"].startswith("REGRESSION_CASE_PLACEHOLDER")]
    if not include_regressions:
        cases = [c for c in cases if not c.get("is_regression")]
    return cases


async def _rebuild_numbered_context(store: QdrantStore, citations: list[dict]) -> str:
    """Re-fetches full chunk text by chunk_id (citation `snippet`s are
    truncated to 280 chars -- not enough to re-judge against) -- same
    pattern scripts/replay_trace.py already uses for the same reason.
    """
    if not citations:
        return ""
    chunk_ids = [c["chunk_id"] for c in citations]
    records = await store.get_by_ids(chunk_ids)
    records_by_id = {str(r.id): r for r in records}
    blocks = []
    for c in citations:
        record = records_by_id.get(c["chunk_id"])
        text = record.payload["text"] if record is not None else "[chunk no longer in Qdrant]"
        blocks.append(f"[{c['marker']}] (source: {c['filename']}, page {c.get('page_number')})\n{text}")
    return "\n\n".join(blocks)


async def generate_raw_answer(
    case: dict, settings: Settings, store: QdrantStore, generator: LLMClient, generator_provider: str,
) -> dict[str, Any]:
    """Runs one case through the real pipeline with skip_judge=True and
    returns a plain-dict record with everything needed later to (a) run
    deterministic assertions, (b) run the judge fresh without
    regenerating the answer, (c) show a human the answer for blind
    labeling. No `judgment` key -- by construction, the judge never ran.
    """
    final: ChatStreamFinal | None = None
    async for event_type, payload in answer_query(
        query=case["question"], use_keyword_search=True, use_query_expansion=False, use_mmr=False, top_k=6,
        doc_ids=None, settings=settings, store=store, generator=generator, judge=generator,
        generator_provider=generator_provider, judge_provider=generator_provider,
        generator_temperature=0.2, judge_temperature=0.0, trace_store=None, langfuse=None, skip_judge=True,
    ):
        if event_type == "final":
            final = payload
    assert final is not None

    citations = [c.model_dump(mode="json") for c in final.citations]
    numbered_context = await _rebuild_numbered_context(store, citations)

    # `final.judgment` can be non-None here EVEN THOUGH skip_judge=True: the
    # no-candidates gate and both off-topic gates in answer_query build
    # their own fixed Judgment (verdict=no_answer/out_of_scope) BEFORE ever
    # reaching the real judge call that skip_judge actually suppresses.
    # Keeping it (rather than discarding it) means the blind labeler and
    # the out_of_jurisdiction_refused assertion both see the true gate
    # outcome, not an artificially blanked one.
    gate_judgment = final.judgment.model_dump(mode="json") if final.judgment is not None else None

    return {
        "id": case["id"], "mode": case["mode"], "checks": case["checks"], "question": case["question"],
        "answer": final.answer, "citations": citations, "numbered_context": numbered_context,
        "retrieval_debug": final.retrieval_debug.model_dump(mode="json"), "judgment": gate_judgment,
    }


def save_raw_answers(answers: list[dict]) -> None:
    _RAW_ANSWERS_PATH.write_text(json.dumps(answers, indent=2), encoding="utf-8")


def load_raw_answers() -> list[dict]:
    if not _RAW_ANSWERS_PATH.exists():
        raise FileNotFoundError(
            f"{_RAW_ANSWERS_PATH} doesn't exist yet -- run scripts/generate_week6_answers.py first."
        )
    return json.loads(_RAW_ANSWERS_PATH.read_text(encoding="utf-8"))


def run_deterministic_checks(raw_answer: dict) -> dict[str, tuple[bool, str]]:
    response_for_checks = {
        "answer": raw_answer["answer"], "citations": raw_answer["citations"],
        "judgment": raw_answer.get("judgment"),
    }
    return {name: ASSERTIONS[name](response_for_checks, raw_answer) for name in raw_answer["checks"]}


async def run_judge_on_answer(raw_answer: dict, judge: LLMClient, temperature: float = 0.0) -> dict:
    judgment = await judge.judge_answer(
        raw_answer["question"], raw_answer["answer"], raw_answer["numbered_context"], temperature,
    )
    return judgment.model_dump(mode="json")


_REFUSAL_CHECK = "out_of_jurisdiction_refused"
LABELS_PATH = Path(__file__).parent / "labels_25.json"


async def grade_one(case: dict, settings: Settings, store: QdrantStore, generator: LLMClient, provider: str) -> dict:
    """Grades one case end-to-end: generate (real pipeline) -> deterministic
    checks -> judge, unless a gate (no-candidates/off-topic) already produced
    its own fixed verdict, in which case that verdict is graded instead of
    calling the judge a second time for no reason. Shared by
    scripts/run_week6_eval.py and the /evals/week6/run API route so how a
    case is graded has exactly one implementation.

    A case PASSES iff every one of its deterministic checks passes AND (when
    a real judge call happened) the judge's verdict is "grounded" -- for a
    case that expects a refusal, a gate firing IS the pass condition instead
    of "grounded" (never double-counted against the wrong target).
    """
    raw = await generate_raw_answer(case, settings, store, generator, provider)
    checks = run_deterministic_checks(raw)
    checks_passed = all(passed for passed, _reason in checks.values())

    expects_refusal = _REFUSAL_CHECK in case["checks"]
    if raw["judgment"] is not None:
        judge_verdict = raw["judgment"]["verdict"]
        judged_by = "gate"
        case_passed = checks_passed and expects_refusal
    else:
        judgment = await run_judge_on_answer(raw, generator)
        judge_verdict = judgment["verdict"]
        judged_by = "judge"
        case_passed = checks_passed and judge_verdict == "grounded"

    return {
        "id": case["id"], "mode": case["mode"], "question": case["question"], "answer": raw["answer"],
        "checks": {name: {"passed": passed, "reason": reason} for name, (passed, reason) in checks.items()},
        "judge_verdict": judge_verdict, "judged_by": judged_by, "passed": case_passed,
    }
