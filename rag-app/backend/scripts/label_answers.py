"""Week 6, step 2: blind human labeling -- YOU read each raw answer (from
scripts/generate_week6_answers.py's output, judge never having run on it)
and decide pass/fail yourself, on the judge's single remaining binary
criterion:

    "Is this answer fully grounded in the retrieved context, with no
     unsupported claims?"  (y = grounded, n = not grounded)

This is the one criterion left in judge_v1.j2 after the 2+ assertable
criteria (section reference, handbook version, notice-period numeric,
out-of-jurisdiction refusal -- see evals/assertions.py) were pulled out
into deterministic code per requirement #2.

The saved file's mtime (and, once you `git add`+`git commit` it, the
commit) is the ordering proof requirement #3 asks for -- commit
labels_25.json BEFORE running scripts/run_week6_eval.py or
scripts/measure_judge_agreement.py against it. This script refuses to run
if week6_results.json (judge output) already exists newer than the raw
answers, as a guard against accidentally labeling with the judge's
verdicts already in view.

Run from backend/, with PYTHONPATH set to it (interactive -- needs a real
terminal, not something to pipe/automate):

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/label_answers.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evals.week6_eval import load_raw_answers

_LABELS_PATH = Path(__file__).parent.parent / "evals" / "labels_25.json"
_CRITERION = "Is this answer fully grounded in the retrieved context, with no unsupported claims?"


def _load_existing_labels() -> dict:
    if _LABELS_PATH.exists():
        return json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    return {"criterion": _CRITERION, "labeled_at_first": None, "labeled_at_last": None, "labels": {}, "notes": {}}


def _save(state: dict) -> None:
    _LABELS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    answers = load_raw_answers()
    state = _load_existing_labels()
    already = set(state["labels"])
    remaining = [a for a in answers if a["id"] not in already]

    if not remaining:
        print(f"All {len(answers)} case(s) already labeled in {_LABELS_PATH}. Delete it to relabel from scratch.")
        return

    print(f"Blind labeling -- {len(remaining)} of {len(answers)} case(s) remaining.")
    print(f"Criterion: {_CRITERION}")
    print("The judge has NOT been run on these yet -- your answer here is not influenced by it.")
    print("Answer y (pass/grounded), n (fail/not grounded), or q to save and quit.\n")

    now = datetime.now(timezone.utc).isoformat()
    if state["labeled_at_first"] is None:
        state["labeled_at_first"] = now

    for i, ans in enumerate(remaining, 1):
        print(f"--- [{i}/{len(remaining)}] {ans['id']}  (mode: {ans['mode']}) ---")
        print(f"Q: {ans['question']}")
        print(f"A: {ans['answer']}")
        if ans["citations"]:
            print("Citations: " + "; ".join(f"[{c['marker']}] {c['filename']} p.{c.get('page_number')}" for c in ans["citations"]))
        else:
            print("Citations: (none)")

        while True:
            choice = input("Grounded? [y/n/q]: ").strip().lower()
            if choice in ("y", "n", "q"):
                break
            print("  please answer y, n, or q")

        if choice == "q":
            break

        state["labels"][ans["id"]] = choice == "y"
        state["labeled_at_last"] = datetime.now(timezone.utc).isoformat()
        _save(state)  # incremental -- a crash or Ctrl+C mid-session doesn't lose earlier labels
        print()

    _save(state)
    done = len(state["labels"])
    print(f"\nSaved {done}/{len(answers)} label(s) to {_LABELS_PATH}.")
    if done >= len(answers):
        print(
            "All cases labeled. Next: commit this file NOW, before running the judge --\n"
            f"  git add {_LABELS_PATH.relative_to(_LABELS_PATH.parent.parent.parent)}\n"
            '  git commit -m "Week 6: blind labels, predate judge run"\n'
            "then: python scripts/run_week6_eval.py"
        )
    else:
        print("Re-run this script to continue labeling the rest.")


if __name__ == "__main__":
    main()
