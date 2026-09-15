"""Week 5 deliverable: a genuinely random, seeded sample of chat traces --
the provable-not-cherry-picked evidence the mentor review wants. Prints the
seed and the sampled trace_ids, ready to paste into notes.md. Running this
twice with the same seed against the same trace file must print the
identical trace_id list.

Run from backend/, with PYTHONPATH set to it:

    cd backend
    $env:PYTHONPATH = "$PWD"
    python scripts/sample_traces.py --seed 42 --n 20
"""
from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.trace.store import TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Random seed -- paste this into notes.md alongside the sampled trace_ids.")
    parser.add_argument("--n", type=int, default=20, help="Sample size (default 20, per the Week 5 rubric).")
    args = parser.parse_args()

    settings = get_settings()
    store = TraceStore(settings.trace_success_log_path, settings.trace_failure_log_path)
    pool = store.read_all()

    if not pool:
        print(
            f"No traces found at {settings.trace_success_log_path} or {settings.trace_failure_log_path}. "
            "Use the app for a while first, then re-run this."
        )
        return

    sample = store.sample_random(args.n, args.seed)

    print(f"Trace pool size: {len(pool)}")
    print(f"Seed: {args.seed}")
    print(f"Sampled {len(sample)} trace(s):\n")
    for trace in sample:
        print(f"  {trace.trace_id}  ({trace.timestamp.isoformat()})  {trace.question!r}")

    print("\nTrace IDs only (paste into notes.md):")
    print([t.trace_id for t in sample])


if __name__ == "__main__":
    main()
