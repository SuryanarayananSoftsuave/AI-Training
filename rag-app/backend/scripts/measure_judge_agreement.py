"""Week 6, step 3 (run twice -- once as "before", once as "after" your
judge_v2.j2 iteration): re-runs the judge against the SAME frozen raw
answers (evals/week6_raw_answers.json) the blind labels in
evals/labels_25.json were written against, and computes agreement.

Frozen on purpose: only the judge call changes between "before" and
"after" runs (via `--prompt-version v1`/`v2` -- see
app/llm/prompts/__init__.py's versioning), not the underlying answers --
otherwise a changed agreement number couldn't be attributed to the prompt
edit specifically.

Usage (from backend/, PYTHONPATH set to it):

    python scripts/measure_judge_agreement.py --tag before --prompt-version v1
    # ... read the printed disagreements, write prediction.txt, edit
    # app/llm/prompts/templates/judge_v2.j2 using 2 of them as few-shot ...
    python scripts/measure_judge_agreement.py --tag after --prompt-version v2

Writes evals/agreement_<tag>.json and prints every disagreement (not just
the count) so you can read them and name who was right, per requirement #4.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.llm.gemini_client import GeminiClient
from app.models.schemas import JudgeVerdict
from evals.week6_eval import load_raw_answers

_LABELS_PATH = Path(__file__).parent.parent / "evals" / "labels_25.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="'before' or 'after' -- names the output file and printed label.")
    parser.add_argument("--prompt-version", default=None, help="e.g. v1 or v2 -- pins judge_v<N>.j2 regardless of the current live version.")
    args = parser.parse_args()

    if not _LABELS_PATH.exists():
        print(f"{_LABELS_PATH} doesn't exist -- run scripts/label_answers.py (and commit it) first.")
        return

    labels_state = json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    labels: dict[str, bool] = labels_state["labels"]
    answers = {a["id"]: a for a in load_raw_answers()}

    settings = get_settings()
    judge = GeminiClient(settings.gemini_api_key, settings.gemini_generator_model, settings.gemini_judge_model)

    missing = [cid for cid in labels if cid not in answers]
    if missing:
        print(f"WARNING: {len(missing)} labeled case(s) have no matching raw answer (stale labels?): {missing}")

    agree = 0
    total = 0
    disagreements = []
    for case_id, human_grounded in labels.items():
        if case_id not in answers:
            continue
        raw = answers[case_id]
        judgment = await judge.judge_answer(
            raw["question"], raw["answer"], raw["numbered_context"], temperature=0.0, prompt_version=args.prompt_version,
        )
        judge_grounded = judgment.verdict == JudgeVerdict.GROUNDED
        total += 1
        if judge_grounded == human_grounded:
            agree += 1
        else:
            disagreements.append({
                "id": case_id, "question": raw["question"], "answer": raw["answer"],
                "human_label": human_grounded, "judge_verdict": judgment.verdict.value,
                "judge_notes": judgment.notes,
            })

    agreement_pct = round(100 * agree / total, 1) if total else 0.0
    print(f"=== Agreement ({args.tag}, prompt_version={args.prompt_version or 'live default'}) ===")
    print(f"{agree}/{total} agree = {agreement_pct}%\n")

    if disagreements:
        print(f"=== {len(disagreements)} disagreement(s) -- read these before iterating ===\n")
        for d in disagreements:
            print(f"[{d['id']}] {d['question']}")
            print(f"  answer: {d['answer'][:200]}")
            print(f"  human said grounded={d['human_label']}  |  judge said verdict={d['judge_verdict']}")
            print(f"  judge notes: {d['judge_notes']}")
            print()
    else:
        print("No disagreements -- 100% agreement on this run.")

    out_path = Path(__file__).parent.parent / "evals" / f"agreement_{args.tag}.json"
    out_path.write_text(json.dumps({
        "tag": args.tag, "prompt_version": args.prompt_version, "agreement_pct": agreement_pct,
        "agree": agree, "total": total, "disagreements": disagreements,
    }, indent=2), encoding="utf-8")
    print(f"Written to {out_path}.")


if __name__ == "__main__":
    asyncio.run(main())
