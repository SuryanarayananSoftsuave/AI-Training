"""Turns raw Week 6 results (and, once available, blind human labels) into
accuracy / precision / recall / F1 numbers -- rather than the bare pass/fail
table run_week6_eval.py prints today.

Two different kinds of check need two different kinds of metric, and this
module keeps them separate instead of forcing one number onto both:

- Deterministic checks (section_reference_valid, handbook_version_cited,
  notice_period_numeric) compare the pipeline's output against fixed ground
  truth (section_index.json, a filename, a regex) with no independent
  predicted-vs-actual pair to classify -- accuracy (pass rate) is the only
  metric that means anything here.
- out_of_jurisdiction_refused IS a real binary classifier: the Week 6 case
  set has both cases that should refuse and cases that shouldn't, so a full
  confusion matrix (precision/recall/F1) applies.
- The judge's grounded/not-grounded verdict can be scored the same way
  against blind human labels (evals/labels_25.json, written by
  scripts/label_answers.py) once they exist.
"""
from __future__ import annotations

from typing import Any

_REFUSAL_CHECK = "out_of_jurisdiction_refused"


def check_accuracy(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Pass rate per deterministic check name, pooled across every case that
    exercised it (a case's `checks` dict only contains the checks it ran).
    """
    outcomes: dict[str, list[bool]] = {}
    for r in results:
        for name, detail in r["checks"].items():
            outcomes.setdefault(name, []).append(detail["passed"])
    return {
        name: {"passed": sum(v), "total": len(v), "accuracy": round(100 * sum(v) / len(v), 1)}
        for name, v in outcomes.items()
    }


def _confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def refusal_confusion(results: list[dict], cases_by_id: dict[str, dict]) -> dict[str, Any]:
    """Ground truth = does this case's own `checks` list name
    out_of_jurisdiction_refused (i.e. it's SUPPOSED to be refused).
    Prediction = did the run's judge_verdict actually come back
    "out_of_scope" -- the same equality assert_out_of_jurisdiction_refused
    itself uses, so this reproduces that check's own definition of a hit
    rather than a looser one.
    """
    tp = fp = fn = tn = 0
    for r in results:
        case = cases_by_id.get(r["id"])
        if case is None:
            continue
        should_refuse = _REFUSAL_CHECK in case["checks"]
        did_refuse = r["judge_verdict"] == "out_of_scope"
        if should_refuse and did_refuse:
            tp += 1
        elif did_refuse:
            fp += 1
        elif should_refuse:
            fn += 1
        else:
            tn += 1
    return _confusion(tp, fp, fn, tn)


def judge_vs_human(results: list[dict], labels: dict[str, bool] | None) -> dict[str, Any] | None:
    """Precision/recall/F1 of the judge's grounded/not-grounded call against
    blind human labels (evals/labels_25.json, written BEFORE the judge ever
    sees the answer -- see scripts/label_answers.py). Returns None -- not
    zeros -- when no labels exist yet, so the caller can tell "no signal
    yet" apart from "measured and it's 0".
    """
    if not labels:
        return None
    results_by_id = {r["id"]: r for r in results}
    tp = fp = fn = tn = 0
    n = 0
    for case_id, human_grounded in labels.items():
        r = results_by_id.get(case_id)
        if r is None:
            continue
        n += 1
        judge_grounded = r["judge_verdict"] == "grounded"
        if judge_grounded and human_grounded:
            tp += 1
        elif judge_grounded:
            fp += 1
        elif human_grounded:
            fn += 1
        else:
            tn += 1
    if n == 0:
        return None
    result = _confusion(tp, fp, fn, tn)
    result["n"] = n
    return result


def build_report(results: list[dict], cases: list[dict], labels: dict[str, bool] | None) -> dict[str, Any]:
    cases_by_id = {c["id"]: c for c in cases}
    overall_pass = sum(1 for r in results if r["passed"])
    return {
        "n_cases": len(results),
        "overall_pass_rate": round(100 * overall_pass / len(results), 1) if results else 0.0,
        "check_accuracy": check_accuracy(results),
        "refusal_confusion": refusal_confusion(results, cases_by_id),
        "judge_vs_human": judge_vs_human(results, labels),
    }
