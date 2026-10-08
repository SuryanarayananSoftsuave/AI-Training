from __future__ import annotations

import threading

import streamlit as st

from utils.api_client import BackendClient


def _get_state() -> dict:
    if "week6_eval_state" not in st.session_state:
        st.session_state.week6_eval_state = {
            "running": False, "stop": False, "stopped": False, "progress": [], "report": None, "error": None,
        }
    return st.session_state.week6_eval_state


def _run_worker(client: BackendClient, provider: str, state: dict) -> None:
    """25 cases is real LLM cost (generate + judge per case) -- `stream`
    is kept as a reference specifically so Stop can call `.close()` on it
    directly: closing the underlying SSE connection makes the backend's
    StreamingResponse detect the client disconnect and cancel whatever
    case is still in flight server-side, not just stop the UI from
    showing further progress. Same pattern as trace_batch.py's per-question stop.
    """
    stream = client.run_week6_eval(provider=provider)
    try:
        for event_type, data in stream:
            if state["stop"]:
                stream.close()
                state["stopped"] = True
                break
            if event_type == "progress":
                state["progress"].append(data)
            elif event_type == "error":
                # A mid-run failure (e.g. a transient LLM provider error) --
                # named which case it hit, instead of just dropping the
                # connection and leaving a generic transport error.
                state["error"] = f"case {data.get('id')}: {data.get('message', 'unknown error')}"
            elif event_type == "final":
                state["report"] = data["report"]
    except Exception as exc:
        state["error"] = str(exc)
    state["running"] = False


def _render_confusion(title: str, caption: str, c: dict) -> None:
    st.markdown(f"**{title}**")
    st.caption(caption)
    cols = st.columns(4)
    cols[0].metric("Precision", f"{c['precision']:.0%}" if c["precision"] is not None else "n/a")
    cols[1].metric("Recall", f"{c['recall']:.0%}" if c["recall"] is not None else "n/a")
    cols[2].metric("F1", f"{c['f1']:.2f}" if c["f1"] is not None else "n/a")
    cols[3].metric("Accuracy", f"{c['accuracy']:.0%}" if c["accuracy"] is not None else "n/a")
    st.caption(f"TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")


def _render_report(report: dict) -> None:
    st.success(f"{report['n_cases']} cases graded — overall pass rate {report['overall_pass_rate']}%")

    st.markdown("**Deterministic check accuracy**")
    st.caption(
        "Pass rate only — these checks compare the pipeline's output to fixed ground truth "
        "(section index, filename, regex), so there's no independent predicted-vs-actual pair "
        "to compute precision/recall from."
    )
    rows = [
        {"check": name, "passed": v["passed"], "total": v["total"], "accuracy": f"{v['accuracy']}%"}
        for name, v in report["check_accuracy"].items()
    ]
    st.table(rows)

    _render_confusion(
        "Out-of-jurisdiction refusal",
        "Ground truth: the case's checks list expects a refusal. Prediction: the run actually returned out_of_scope.",
        report["refusal_confusion"],
    )

    if report["judge_vs_human"] is not None:
        _render_confusion(
            "Judge grounded-verdict vs. blind human labels",
            f"n={report['judge_vs_human']['n']} cases labeled in evals/labels_25.json.",
            report["judge_vs_human"],
        )
    else:
        st.info(
            "Judge-vs-human precision/recall isn't available yet — from backend/, run "
            "`python scripts/label_answers.py` to blind-label the answers first "
            "(writes evals/labels_25.json), then run this eval again."
        )


@st.fragment(run_every=1)
def _live_eval_panel(client: BackendClient, provider: str) -> None:
    state = _get_state()

    if state["running"]:
        progress = state["progress"]
        total = progress[-1]["total"] if progress else None
        done = sum(1 for p in progress if p["status"] == "done")
        if total:
            st.progress(done / total, text=f"Running... {done}/{total} cases graded (real LLM calls, a few minutes)")
        else:
            st.caption("Starting...")
        if st.button("⏹ Stop", key="stop_week6_eval", type="primary"):
            state["stop"] = True
    else:
        if st.button("▶ Run Week 6 eval" if state["report"] is None else "▶ Run again", key="run_week6_eval"):
            state["running"] = True
            state["stop"] = False
            state["stopped"] = False
            state["progress"] = []
            state["report"] = None
            state["error"] = None
            threading.Thread(target=_run_worker, args=(client, provider, state), daemon=True).start()
            st.rerun(scope="fragment")

    if state["stopped"]:
        done = sum(1 for p in state["progress"] if p["status"] == "done")
        st.warning(
            f"Stopped after {done} case(s) -- the case already in progress when you clicked Stop finished, "
            "but no further case was started. No report shown for a partial run (pass-rate-by-mode needs the full set)."
        )

    if state["error"]:
        st.error(f"Eval run failed: {state['error']}")

    if state["report"] is not None:
        _render_report(state["report"])


def render_week6_eval_panel(client: BackendClient) -> None:
    st.subheader("📊 Automated Eval")
    st.caption(
        "Re-runs all Week 6 cases through the live pipeline (real LLM calls) and scores them: "
        "pass rate per deterministic check, plus a precision/recall/F1 confusion matrix for the "
        "refusal check and, once blind human labels exist, for the judge's grounded verdict."
    )
    provider = st.selectbox("Provider", ["gemini", "groq"], key="week6_eval_provider")
    _live_eval_panel(client, provider)
