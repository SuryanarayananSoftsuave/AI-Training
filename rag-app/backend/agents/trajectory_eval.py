"""Week 8 (W8-Task-Set-C.md): trajectory scoring for the ReAct agent
(agents/react_agent.py). Shared by scripts/run_trajectory_eval.py and
POST /agents/trajectory/run -- exactly one implementation, two ways to
trigger it, same convention as evals/week6_eval.py::grade_one being shared
between run_week6_eval.py and /evals/week6/run.

Deliberately does NOT reuse agents/fixtures/employee_questions.json (Week
7's 10 questions): verdict.txt documents that all 10 force an identical
2-tool path regardless of input, which would manufacture a fake 0%
outcome-vs-trajectory gap. See agents/fixtures/trajectory_cases.json for
the real 10 cases, including 2 multi-turn conversations built to exercise
the actual gap in agents/conversation.py -- jurisdiction persists across
turns, tenure_years never does.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from app.core.config import Settings
from app.registry.session_store import SessionStore
from app.retrieval.qdrant_store import QdrantStore
from groq import AsyncGroq

from agents.conversation import ConversationState, Turn, append_turn
from agents.employees import EMPLOYEES
from agents.react_agent import Budgets, StepRecord, run_agent

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CASES_PATH = _FIXTURES_DIR / "trajectory_cases.json"
_RESULTS_DIR = Path(__file__).parent

# react_agent.py's default 30s wall-clock budget is genuinely borderline for
# any case that calls search_handbook -- real retrieval+reranking alone
# takes ~15-20s, on top of normal LLM turnaround. Observed live: t07 (the
# one case testing section-number citation) hit that ceiling before it
# could write a final answer, which made section_number_validity_rate
# report "n/a" for a timing reason, not because the check itself is broken.
# A trajectory-eval-specific, more generous ceiling fixes the measurement
# without touching react_agent.py's default (Week 7's agent race, the
# dispatcher, and the conversation endpoint all still use the tighter
# default, which is deliberately part of THEIR story about budget
# enforcement -- this constant only affects this file's own calls).
_TRAJECTORY_BUDGETS = Budgets(max_wall_clock_s=60.0)

_SECTION_CITATION_RE = re.compile(r"(?:section|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_NOTICE_PERIOD_RE = re.compile(r"\b(\d+)\s*(day|days|week|weeks)\b", re.IGNORECASE)
_SECTION_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "sample_data" / "hr_policy_kb_md" / "section_index.json"


def load_cases() -> list[dict]:
    return json.loads(_CASES_PATH.read_text(encoding="utf-8"))["cases"]


def _load_section_index() -> set[tuple[str, str]]:
    data = json.loads(_SECTION_INDEX_PATH.read_text(encoding="utf-8"))
    return {(s["doc"], s["section_id"]) for s in data["sections"]}


@dataclass
class CaseResult:
    id: str
    question: str  # the SCORED turn's question text (turn 2, for conversation cases)
    answer: str | None
    steps: list[StepRecord]
    terminated_reason: str | None
    total_tokens: int  # summed across BOTH turns for conversation cases
    cost_usd: float  # summed across BOTH turns
    wall_clock_s: float  # scored turn only
    tool_choice_passed: bool
    tool_choice_detail: str
    employee_id_checked: int
    employee_id_valid: int
    section_checked: int
    section_valid: int
    argument_issues: list[str]
    step_efficiency: float | None  # None for diagnostic-only cases
    outcome_passed: bool | None  # None when the case has no single checkable answer
    diagnostic_only: bool = False
    zoo_modes: list[str] = field(default_factory=list)
    # Set only when the underlying agent run itself failed (a transient
    # provider error, a network issue, etc.) -- distinct from a case that
    # completed but scored badly. Every other field is left at its most
    # conservative "unscored" value when this is set (see run_case).
    error: str | None = None

    @property
    def argument_checked(self) -> int:
        """Pooled total across both argument-realism axes -- kept for a
        quick overview number; build_report also reports
        employee_id_validity_rate/section_number_validity_rate separately
        so one broken axis can't hide behind the other's volume.
        """
        return self.employee_id_checked + self.section_checked

    @property
    def argument_valid(self) -> int:
        return self.employee_id_valid + self.section_valid


def _matches_lookup_before_rule_per_employee(steps: list[StepRecord], employee_ids: list[str]) -> bool:
    """The one 'set, not a sequence' rule (case t05): EACH employee's own
    get_employee_record call must occur before THAT SAME employee's own
    get_notice_period_rule call -- but which employee is handled first is
    unconstrained, so multiple concrete orderings are all correct.

    get_notice_period_rule takes (jurisdiction, tenure_years), not an
    employee_id, so a rule call is attributed back to a specific employee
    by matching that signature against EMPLOYEES -- this only disambiguates
    correctly when the case's employees don't share an identical
    (jurisdiction, tenure_years) pair, true for every case in
    trajectory_cases.json today.
    """
    lookup_index: dict[str, int] = {}
    for i, s in enumerate(steps):
        if s.action == "get_employee_record" and s.action_input.get("employee_id") in employee_ids:
            lookup_index.setdefault(s.action_input["employee_id"], i)
    if set(lookup_index) != set(employee_ids):
        return False

    for eid in employee_ids:
        employee = EMPLOYEES.get(eid)
        if employee is None:
            return False
        signature = (employee.jurisdiction, employee.tenure_years)
        rule_index = next(
            (
                i for i, s in enumerate(steps)
                if s.action == "get_notice_period_rule"
                and (s.action_input.get("jurisdiction"), s.action_input.get("tenure_years")) == signature
            ),
            None,
        )
        if rule_index is None or rule_index <= lookup_index[eid]:
            return False
    return True


_SEQUENCE_RULES: dict[str, Callable[[list[StepRecord], list[str]], bool]] = {
    "lookup_before_rule_per_employee": _matches_lookup_before_rule_per_employee,
}


def _score_tool_choice(case: dict, steps: list[StepRecord]) -> tuple[bool, str]:
    actual = [s.action for s in steps]
    if case.get("diagnostic_only"):
        return True, f"diagnostic-only case, not scored for tool-choice accuracy (actual: {actual})"
    if "sequence_rule" in case:
        rule = _SEQUENCE_RULES[case["sequence_rule"]]
        passed = rule(steps, case["relevant_employee_ids"])
        return passed, f"sequence_rule={case['sequence_rule']} -> {'pass' if passed else 'fail'} (actual: {actual})"
    passed = actual in case["expected_sequences"]
    return passed, f"actual {actual} {'in' if passed else 'NOT in'} expected {case['expected_sequences']}"


def _score_arguments(case: dict, steps: list[StepRecord], answer: str | None) -> tuple[int, int, int, int, list[str]]:
    """Returns (employee_id_checked, employee_id_valid, section_checked,
    section_valid, issue_descriptions) -- two INDEPENDENT axes, per the
    rubric's own phrasing ("employee ids and policy section numbers"),
    reported separately rather than pooled so one broken axis can't hide
    behind the other's volume.

    Section-number extraction requires a "section"/"§" keyword immediately
    before the digits (`_SECTION_CITATION_RE`), not just the first bare
    number anywhere in the free-text answer -- an earlier, unanchored
    version grabbed e.g. "3" out of "3 days per week" instead of "3.1" out
    of "see section 3.1", which is both a false-PASS risk (a coincidentally
    real section id) and a false-FAIL risk (an unrelated number that
    resolves to nothing). The extracted number is then paired with
    whichever document(s) search_handbook actually searched this turn (its
    own `source` observation), matching evals/assertions.py's
    (doc, section_id) pairing instead of checking the bare number against
    every document's section ids.
    """
    employee_checked = 0
    employee_valid = 0
    issues: list[str] = []
    for s in steps:
        if s.action == "get_employee_record":
            employee_checked += 1
            eid = s.action_input.get("employee_id")
            if eid in EMPLOYEES:
                employee_valid += 1
            else:
                issues.append(f"hallucinated employee_id {eid!r} (not in EMPLOYEES)")

    section_checked = 0
    section_valid = 0
    if case.get("expects_section_citation") and answer:
        match = _SECTION_CITATION_RE.search(answer)
        if match:
            section_checked += 1
            cited_number = match.group(1)
            searched_docs: set[str] = set()
            for s in steps:
                if s.action == "search_handbook":
                    try:
                        source = json.loads(s.observation).get("source")
                    except (json.JSONDecodeError, AttributeError):
                        source = None
                    if source:
                        searched_docs.add(Path(source).stem)
            index = _load_section_index()
            if searched_docs:
                resolves = any((doc, cited_number) in index for doc in searched_docs)
            else:
                resolves = any(number == cited_number for _doc, number in index)
            if resolves:
                section_valid += 1
            else:
                issues.append(f"cited section number {cited_number!r} does not resolve to any real indexed section")

    return employee_checked, employee_valid, section_checked, section_valid, issues


def _has_numeric_notice_period(text: str | None) -> bool:
    return bool(text) and bool(_NOTICE_PERIOD_RE.search(text))


def _score_outcome(case: dict, answer: str | None) -> bool | None:
    if case.get("expected_days_by_employee"):
        if not answer:
            return False
        return all(str(days) in answer for days in case["expected_days_by_employee"].values())
    if case.get("expected_days") is not None:
        return answer is not None and str(case["expected_days"]) in answer
    return None  # no single correct answer to check (t06, t07)


def _detect_zoo_modes(
    case: dict, steps: list[StepRecord], answer: str | None, terminated_reason: str | None,
    tool_choice_passed: bool, argument_issues: list[str],
) -> list[str]:
    modes: list[str] = []
    if terminated_reason is not None:
        modes.append("loop_or_budget_blowout")
    if not tool_choice_passed and not case.get("diagnostic_only"):
        modes.append("wrong_tool_or_extra_call")
    if any("hallucinated employee_id" in i for i in argument_issues):
        modes.append("hallucinated_employee_id")
    if any("does not resolve to any real indexed section" in i for i in argument_issues):
        modes.append("fluent_fiction_section_number")
    relevant_ids = case.get("relevant_employee_ids") or []
    looked_up = {s.action_input.get("employee_id") for s in steps if s.action == "get_employee_record"}
    if relevant_ids and not set(relevant_ids).issubset(looked_up) and _has_numeric_notice_period(answer):
        modes.append("guessed_without_reading_tenure")
    # `expected_days_by_employee` (case t05, the comparison case) is a
    # SEPARATE checkable-answer field from `expected_days` -- checking only
    # the latter meant this mode could never fire for t05 even when the
    # agent genuinely gave up on the one case built to exercise multi-
    # employee branching.
    has_checkable_answer = case.get("expected_days") is not None or bool(case.get("expected_days_by_employee"))
    if terminated_reason is None and has_checkable_answer and not _has_numeric_notice_period(answer):
        modes.append("gave_up_quietly")
    return modes


async def run_case(
    case: dict, settings: Settings, store: QdrantStore, groq_api_key: str, model: str,
    mitigation: bool, session_store: SessionStore | None = None,
) -> CaseResult:
    """Runs one case, never raising: a failure in the underlying agent run
    itself (a transient Groq/network error, etc.) is caught here and
    returned as a CaseResult with `error` set, rather than propagating and
    losing every other case's already-completed results in the same batch
    -- one bad case out of 10 used to discard the whole run.
    """
    try:
        return await _run_case_impl(case, settings, store, groq_api_key, model, mitigation, session_store)
    except Exception as exc:
        logger.warning("Week 8 trajectory case %s failed: %s", case["id"], exc)
        question_text = case["question"] if case["type"] != "conversation" else case["turns"][-1]
        return CaseResult(
            id=case["id"], question=question_text, answer=None, steps=[], terminated_reason=None,
            total_tokens=0, cost_usd=0.0, wall_clock_s=0.0,
            tool_choice_passed=False, tool_choice_detail=f"case errored before scoring: {exc}",
            employee_id_checked=0, employee_id_valid=0, section_checked=0, section_valid=0,
            argument_issues=[], step_efficiency=None, outcome_passed=None,
            diagnostic_only=bool(case.get("diagnostic_only")), zoo_modes=["case_errored"], error=str(exc),
        )


async def _run_case_impl(
    case: dict, settings: Settings, store: QdrantStore, groq_api_key: str, model: str,
    mitigation: bool, session_store: SessionStore | None = None,
) -> CaseResult:
    """Runs one case through the real agent -- a single question, or a
    2-turn conversation for `type == "conversation"` -- and scores the
    SCORED turn (the last one) against its trajectory expectations.
    `mitigation` toggles react_agent.py's require_employee_lookup gate, the
    one code-level mitigation this eval measures before -> after.
    """
    total_tokens = 0
    total_cost = 0.0

    if case["type"] == "conversation":
        client = AsyncGroq(api_key=groq_api_key, max_retries=6)
        state = ConversationState(session_id=f"trajectory-{case['id']}")
        turn1_question, turn2_question = case["turns"]

        seed_result = await run_agent(
            # Turn 1 is only ever used to SEED conversation state -- it is
            # never scored. Applying the mitigation here too would let its
            # correction cost (an extra lap/tokens) leak into the "before
            # vs after" cost comparison for a reason unrelated to what's
            # being measured (protecting the SCORED turn 2 answer).
            turn1_question, settings, store, groq_api_key, model, _TRAJECTORY_BUDGETS,
            conversation=state, session_store=session_store, require_employee_lookup=False,
        )
        total_tokens += seed_result.total_tokens
        total_cost += seed_result.cost_usd
        await append_turn(
            state, Turn(question=turn1_question, answer=seed_result.answer), client, model,
            settings.conversation_window_turns, settings.conversation_fold_batch_size,
        )

        scored = await run_agent(
            turn2_question, settings, store, groq_api_key, model, _TRAJECTORY_BUDGETS,
            conversation=state, session_store=session_store, require_employee_lookup=mitigation,
        )
        question_text = turn2_question
    else:
        scored = await run_agent(
            case["question"], settings, store, groq_api_key, model, _TRAJECTORY_BUDGETS, require_employee_lookup=mitigation,
        )
        question_text = case["question"]

    total_tokens += scored.total_tokens
    total_cost += scored.cost_usd

    tool_choice_passed, tool_choice_detail = _score_tool_choice(case, scored.steps)
    employee_id_checked, employee_id_valid, section_checked, section_valid, argument_issues = _score_arguments(
        case, scored.steps, scored.answer,
    )
    outcome_passed = _score_outcome(case, scored.answer)

    min_steps = case.get("min_steps_needed") or 0
    diagnostic_only = bool(case.get("diagnostic_only"))
    if diagnostic_only:
        step_efficiency = None
    elif min_steps > 0:
        step_efficiency = len(scored.steps) / min_steps
    else:
        step_efficiency = float(len(scored.steps))  # any call at all is already "extra" for a 0-steps-needed case

    zoo_modes = _detect_zoo_modes(case, scored.steps, scored.answer, scored.terminated_reason, tool_choice_passed, argument_issues)

    return CaseResult(
        id=case["id"], question=question_text, answer=scored.answer, steps=scored.steps,
        terminated_reason=scored.terminated_reason, total_tokens=total_tokens, cost_usd=total_cost,
        wall_clock_s=scored.wall_clock_s, tool_choice_passed=tool_choice_passed, tool_choice_detail=tool_choice_detail,
        employee_id_checked=employee_id_checked, employee_id_valid=employee_id_valid,
        section_checked=section_checked, section_valid=section_valid, argument_issues=argument_issues,
        step_efficiency=step_efficiency, outcome_passed=outcome_passed, diagnostic_only=diagnostic_only,
        zoo_modes=zoo_modes,
    )


async def run_all_cases(
    settings: Settings, store: QdrantStore, groq_api_key: str, model: str, mitigation: bool,
    session_store: SessionStore | None = None,
    on_progress: Callable[[dict, CaseResult], Awaitable[None]] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in load_cases():
        result = await run_case(case, settings, store, groq_api_key, model, mitigation, session_store)
        results.append(result)
        if on_progress is not None:
            await on_progress(case, result)
    return results


def _case_out(r: CaseResult) -> dict:
    return {
        "id": r.id, "question": r.question, "answer": r.answer,
        "steps": [asdict(s) for s in r.steps],
        "terminated_reason": r.terminated_reason,
        "total_tokens": r.total_tokens, "cost_usd": round(r.cost_usd, 6), "wall_clock_s": round(r.wall_clock_s, 2),
        "tool_choice_passed": r.tool_choice_passed, "tool_choice_detail": r.tool_choice_detail,
        "employee_id_checked": r.employee_id_checked, "employee_id_valid": r.employee_id_valid,
        "section_checked": r.section_checked, "section_valid": r.section_valid,
        "argument_checked": r.argument_checked, "argument_valid": r.argument_valid, "argument_issues": r.argument_issues,
        "step_efficiency": r.step_efficiency, "outcome_passed": r.outcome_passed, "zoo_modes": r.zoo_modes,
        "diagnostic_only": r.diagnostic_only, "error": r.error,
    }


def build_report(results: list[CaseResult]) -> dict:
    """The 4 required numbers (tool-choice accuracy, argument validity,
    step efficiency, cost p50+max), the outcome-vs-trajectory gap, one
    named right-answer-wrong-path example, and per-mode failure counts --
    mirrors evals/metrics.py::build_report's shape/spirit for Week 6.
    """
    # Excludes both diagnostic-only cases AND errored ones -- an errored
    # case (a transient provider/network failure) is a different kind of
    # non-result than a real trajectory judgment, and mixing "the agent
    # made a bad choice" with "the connection dropped" into one denominator
    # would make the before/after mitigation comparison noisy for reasons
    # that have nothing to do with the mitigation.
    scorable = [r for r in results if not r.diagnostic_only and r.error is None]
    n_scorable = len(scorable) or 1
    n_errored = sum(1 for r in results if r.error is not None)

    tool_choice_accuracy = round(100 * sum(r.tool_choice_passed for r in scorable) / n_scorable, 1)

    # Reported as two INDEPENDENT rates (the rubric names both axes
    # separately: "were the employee ids AND policy section numbers real
    # or fluent fiction?") -- pooling them into one number let a broken
    # section-number check hide behind the much larger volume of clean
    # employee-id checks. Both now use `scorable` (excludes the
    # diagnostic-only case), consistent with tool_choice_accuracy and
    # outcome_pass_rate below, rather than the full `results`.
    employee_checked_total = sum(r.employee_id_checked for r in scorable)
    employee_valid_total = sum(r.employee_id_valid for r in scorable)
    employee_id_validity_rate = (
        round(100 * employee_valid_total / employee_checked_total, 1) if employee_checked_total else None
    )

    section_checked_total = sum(r.section_checked for r in scorable)
    section_valid_total = sum(r.section_valid for r in scorable)
    section_number_validity_rate = (
        round(100 * section_valid_total / section_checked_total, 1) if section_checked_total else None
    )

    total_checked = employee_checked_total + section_checked_total
    total_valid = employee_valid_total + section_valid_total
    argument_validity_rate = round(100 * total_valid / total_checked, 1) if total_checked else None

    efficiencies = [r.step_efficiency for r in results if r.step_efficiency is not None]
    step_efficiency_avg = round(sum(efficiencies) / len(efficiencies), 2) if efficiencies else None

    # Errored cases are excluded here too -- their cost_usd is a hardcoded
    # 0.0 (we don't have a reliable partial-cost figure at the point of
    # failure), and including that $0 would silently pull p50/max down
    # rather than honestly reflect "we don't know."
    costs = [r.cost_usd for r in results if r.error is None]
    cost_p50 = round(statistics.median(costs), 6) if costs else 0.0
    cost_max = round(max(costs), 6) if costs else 0.0

    outcome_scorable = [r for r in scorable if r.outcome_passed is not None]
    outcome_pass_rate = round(100 * sum(r.outcome_passed for r in outcome_scorable) / len(outcome_scorable), 1) if outcome_scorable else None
    gap = round(outcome_pass_rate - tool_choice_accuracy, 1) if outcome_pass_rate is not None else None

    named_example = None
    for r in outcome_scorable:
        if r.outcome_passed and not r.tool_choice_passed:
            named_example = _case_out(r)
            break

    mode_counts: dict[str, int] = {}
    for r in results:
        for mode in r.zoo_modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "n_cases": len(results),
        "n_scorable": len(scorable),
        "n_errored": n_errored,
        "tool_choice_accuracy": tool_choice_accuracy,
        "argument_validity_rate": argument_validity_rate,
        "argument_checks_total": total_checked,
        "employee_id_validity_rate": employee_id_validity_rate,
        "employee_id_checks_total": employee_checked_total,
        "section_number_validity_rate": section_number_validity_rate,
        "section_number_checks_total": section_checked_total,
        "step_efficiency_avg": step_efficiency_avg,
        "cost_p50_usd": cost_p50,
        "cost_max_usd": cost_max,
        "outcome_pass_rate": outcome_pass_rate,
        "gap": gap,
        "named_right_answer_wrong_path_example": named_example,
        "mode_counts": mode_counts,
        "cases": [_case_out(r) for r in results],
    }


def regression_diff(before_report: dict, after_report: dict) -> dict:
    """Per-mode count before vs after, flagging any mode that got worse or
    newly appeared -- requirement #5, run across EVERY mode, not just the
    one mitigation targeted.
    """
    before_counts = before_report["mode_counts"]
    after_counts = after_report["mode_counts"]
    rows = []
    for mode in sorted(set(before_counts) | set(after_counts)):
        b, a = before_counts.get(mode, 0), after_counts.get(mode, 0)
        flag = "NEW" if b == 0 and a > 0 else ("WORSE" if a > b else ("IMPROVED" if a < b else "unchanged"))
        rows.append({"mode": mode, "before": b, "after": a, "delta": a - b, "flag": flag})
    return {"rows": rows}


def results_path(tag: str) -> Path:
    return _RESULTS_DIR / f"trajectory_results_{tag}.json"


def save_report(report: dict, tag: str) -> None:
    results_path(tag).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def load_report(tag: str) -> dict | None:
    path = results_path(tag)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
