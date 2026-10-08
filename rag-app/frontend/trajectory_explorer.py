from __future__ import annotations

import json
import threading

import streamlit as st

from agent_view import render_step_trace
from utils.api_client import BackendClient


def _get_state() -> dict:
    if "trajectory_state" not in st.session_state:
        st.session_state.trajectory_state = {
            "running_before": False, "running_after": False,
            "stop_before": False, "stop_after": False,
            "stopped_before": False, "stopped_after": False,
            "progress_before": [], "progress_after": [],
            "report_before": None, "report_after": None,
            # Split per-phase -- a single shared "error" key meant "before"
            # failing displayed under BOTH phases, and starting either
            # phase silently wiped the other phase's still-unresolved error.
            "error_before": None, "error_after": None,
        }
    return st.session_state.trajectory_state


def _run_worker(
    client: BackendClient, mitigation: bool, state: dict, progress_key: str, report_key: str, running_key: str,
    error_key: str, stop_key: str, stopped_key: str,
) -> None:
    """Each case is a real Groq call (or two, for the conversation cases) --
    `stream` is kept as a reference specifically so Stop can call
    `.close()` on it: closing the SSE connection makes the backend's
    StreamingResponse detect the disconnect and cancel whatever case is
    still in flight server-side, not just stop the UI from showing further
    progress. Same pattern as trace_batch.py's per-question stop.
    """
    stream = client.run_trajectory_eval(mitigation=mitigation)
    try:
        for event_type, data in stream:
            if state[stop_key]:
                stream.close()
                state[stopped_key] = True
                break
            if event_type == "progress":
                state[progress_key].append(data)
            elif event_type == "error":
                state[error_key] = f"case {data.get('id')}: {data.get('message', 'unknown error')}"
            elif event_type == "final":
                state[report_key] = data["report"]
    except Exception as exc:
        state[error_key] = str(exc)
    state[running_key] = False


def _render_report(report: dict) -> None:
    if report.get("n_errored"):
        st.warning(
            f"⚠ {report['n_errored']}/{report['n_cases']} case(s) failed to complete (a transient provider/network "
            "error, not a trajectory judgment) and are excluded from every number below -- see which ones in "
            "Per-case detail."
        )

    cols = st.columns(3)
    cols[0].metric("Tool-choice accuracy", f"{report['tool_choice_accuracy']}%")
    cols[1].metric(
        "Argument validity (pooled)",
        f"{report['argument_validity_rate']}%" if report["argument_validity_rate"] is not None else "n/a",
        help=f"{report['argument_checks_total']} argument(s) checked -- see the two axes below, a broken one can hide here",
    )
    cols[2].metric("Step efficiency (avg)", report["step_efficiency_avg"])

    # Cost gets its own row, split into two metrics -- cramming both
    # "$0.00030 / $0.00060" into one narrow 4-column metric box was
    # visually truncating to "$0.00..." even though the underlying numbers
    # were always correct (a display-width issue, not a calculation one).
    cost_cols = st.columns(2)
    cost_cols[0].metric("Cost (p50)", f"${report['cost_p50_usd']:.5f}")
    cost_cols[1].metric("Cost (max)", f"${report['cost_max_usd']:.5f}")

    # The rubric asks about TWO independent kinds of argument realism
    # ("employee ids and policy section numbers") -- shown separately so
    # one broken axis can't hide behind the other's volume in the pooled
    # number above.
    axis_cols = st.columns(2)
    axis_cols[0].metric(
        "↳ Employee-id validity",
        f"{report['employee_id_validity_rate']}%" if report["employee_id_validity_rate"] is not None else "n/a",
        help=f"{report['employee_id_checks_total']} employee_id argument(s) checked",
    )
    axis_cols[1].metric(
        "↳ Section-number validity",
        f"{report['section_number_validity_rate']}%" if report["section_number_validity_rate"] is not None else "n/a",
        help=f"{report['section_number_checks_total']} section citation(s) checked",
    )

    st.metric(
        "Outcome-vs-trajectory gap",
        f"{report['gap']} pts" if report["gap"] is not None else "n/a",
        help="Outcome pass rate minus tool-choice (trajectory) pass rate.",
    )

    example = report.get("named_right_answer_wrong_path_example")
    if example:
        st.markdown("**Right answer, wrong path:**")
        with st.container(border=True):
            st.markdown(f"`{example['id']}` — {example['question']}")
            st.caption(example["tool_choice_detail"])
            st.markdown(f"**Answer:** {example['answer']}")
            render_step_trace(example["steps"])
    else:
        st.caption("No right-answer-wrong-path case found in this run.")

    if report["mode_counts"]:
        st.markdown("**Failure modes detected**")
        st.table([{"mode": k, "count": v} for k, v in report["mode_counts"].items()])
    else:
        st.caption("No failure modes detected in this run.")

    with st.expander(f"Per-case detail ({report['n_cases']} cases)"):
        for c in report["cases"]:
            with st.container(border=True):
                if c.get("error"):
                    st.markdown(f"`{c['id']}` — {c['question']}")
                    st.error(f"Errored before scoring: {c['error']}")
                    continue
                outcome = "n/a" if c["outcome_passed"] is None else ("PASS" if c["outcome_passed"] else "FAIL")
                st.markdown(f"`{c['id']}` — {c['question']}")
                st.caption(
                    f"tool_choice={'PASS' if c['tool_choice_passed'] else 'FAIL'} · outcome={outcome} · "
                    f"steps={len(c['steps'])} · tokens={c['total_tokens']} · cost=${c['cost_usd']:.6f}"
                )
                if c["zoo_modes"]:
                    st.caption(f"⚠️ modes: {', '.join(c['zoo_modes'])}")
                render_step_trace(c["steps"])


def _render_phase(client: BackendClient, mitigation: bool, state: dict) -> None:
    running_key = "running_after" if mitigation else "running_before"
    progress_key = "progress_after" if mitigation else "progress_before"
    report_key = "report_after" if mitigation else "report_before"
    error_key = "error_after" if mitigation else "error_before"
    stop_key = "stop_after" if mitigation else "stop_before"
    stopped_key = "stopped_after" if mitigation else "stopped_before"
    label = "after (mitigation ON)" if mitigation else "before (mitigation OFF)"

    if state[running_key]:
        progress = state[progress_key]
        total = progress[-1]["total"] if progress else None
        done = sum(1 for p in progress if p["status"] in ("done", "errored"))
        errored = sum(1 for p in progress if p["status"] == "errored")
        if total:
            suffix = f", {errored} errored" if errored else ""
            st.progress(done / total, text=f"Running {label}... {done}/{total} cases (real Groq calls){suffix}")
        else:
            st.caption("Starting...")
        if st.button(f"⏹ Stop ({label})", key=f"stop_trajectory_{mitigation}", type="primary"):
            state[stop_key] = True
    else:
        disabled = mitigation and state["report_before"] is None
        btn_label = f"▶ Run live ({label})" if state[report_key] is None else f"▶ Run again ({label})"
        if st.button(btn_label, key=f"run_trajectory_{mitigation}", disabled=disabled):
            state[running_key] = True
            state[stop_key] = False
            state[stopped_key] = False
            state[progress_key] = []
            state[report_key] = None
            state[error_key] = None
            threading.Thread(
                target=_run_worker,
                args=(client, mitigation, state, progress_key, report_key, running_key, error_key, stop_key, stopped_key),
                daemon=True,
            ).start()
            st.rerun(scope="fragment")
        if disabled:
            st.caption("Run 'before' first.")

    if state[stopped_key]:
        done = sum(1 for p in state[progress_key] if p["status"] == "done")
        st.warning(
            f"Stopped ({label}) after {done} case(s) -- the case already in progress when you clicked Stop "
            "finished, but no further case was started. No report shown for a partial run."
        )

    if state[error_key]:
        st.error(f"Run failed: {state[error_key]}")

    if state[report_key] is not None:
        _render_report(state[report_key])


@st.fragment(run_every=1)
def _live_panel(client: BackendClient) -> None:
    state = _get_state()

    st.markdown("### Before mitigation")
    _render_phase(client, False, state)

    st.divider()
    st.markdown("### After mitigation (require_employee_lookup gate)")
    _render_phase(client, True, state)

    if state["report_before"] is not None and state["report_after"] is not None:
        st.divider()
        st.markdown("### Regression check (every mode, before → after)")
        before_counts = state["report_before"]["mode_counts"]
        after_counts = state["report_after"]["mode_counts"]
        rows = []
        for mode in sorted(set(before_counts) | set(after_counts)):
            b, a = before_counts.get(mode, 0), after_counts.get(mode, 0)
            flag = "NEW" if b == 0 and a > 0 else ("WORSE" if a > b else ("IMPROVED" if a < b else "unchanged"))
            rows.append({"mode": mode, "before": b, "after": a, "flag": flag})
        st.table(rows)
        st.caption(
            f"Price paid: cost p50 ${state['report_before']['cost_p50_usd']:.6f} → "
            f"${state['report_after']['cost_p50_usd']:.6f}, "
            f"cost max ${state['report_before']['cost_max_usd']:.6f} → ${state['report_after']['cost_max_usd']:.6f}."
        )


def render_trajectory_eval_panel(client: BackendClient) -> None:
    """Week 8 (W8-Task-Set-C.md): find the outcome-vs-trajectory gap in the
    HR agent, then close one failure mode. Every number here is computed
    from a real agent run made when you click -- nothing is replayed or
    precomputed -- via /agents/trajectory/run (agents/trajectory_eval.py).
    """
    st.subheader("🧭 Trajectory Eval")
    st.caption(
        "Runs the 10 trajectory cases through the real agent live: tool-choice accuracy, "
        "argument validity, step efficiency, and cost p50/max, plus the outcome-vs-trajectory "
        "gap and every failure mode detected. Run 'before', then 'after' to see the one "
        "mitigation's measured effect -- both are genuine live runs, not a stored report."
    )

    try:
        existing = client.get_trajectory_results()
    except Exception as exc:
        existing = None
        st.error(f"Could not reach backend: {exc}")

    if existing and (existing.get("before") or existing.get("after")):
        with st.expander("📄 Already-run results (from disk)", expanded=False):
            if existing.get("before"):
                st.markdown("**Before**")
                _render_report(existing["before"])
            if existing.get("after"):
                st.markdown("**After**")
                _render_report(existing["after"])
            if existing.get("regression"):
                st.markdown("**Regression (before → after)**")
                st.table(existing["regression"]["rows"])

    _live_panel(client)


def render_injection_playground_panel(client: BackendClient) -> None:
    """Week 8 bonus (W8-Task-Set-C.md Sec.5): a real, live attack against
    the real agent -- E1's poisoned manager_comment (agents/employees.py)
    is only ever surfaced through this endpoint, so it never affects the
    graded trajectory eval above. Attack, defend (sanitize), re-attack --
    every click is a genuine agent run, never a canned example.
    """
    st.subheader("🔓 Injection Playground")
    st.caption(
        "Bonus: E1's employee record carries a hidden instruction in a free-text manager_comment "
        "field. This agent has no approve/write tool, so the realistic damage is a corrupted "
        "notice-period answer, not an unauthorized approval. Attack, then defend, then re-attack."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔴 Attack (no defense)", key="injection_attack_raw"):
            try:
                st.session_state["injection_raw_result"] = client.run_injection_attack(sanitize=False)
            except Exception as exc:
                st.error(f"Attack failed: {exc}")
    with col2:
        if st.button("🟢 Re-attack with defense (sanitize)", key="injection_attack_defended"):
            try:
                st.session_state["injection_defended_result"] = client.run_injection_attack(sanitize=True)
            except Exception as exc:
                st.error(f"Attack failed: {exc}")

    for label, key in [("Raw attack", "injection_raw_result"), ("Defended re-attack", "injection_defended_result")]:
        result = st.session_state.get(key)
        if result:
            st.markdown(f"**{label}**")
            st.caption(f"Question asked: {result['question']}")

            # Pull the poisoned field out of step 1's observation and show it
            # on its own line -- the raw vs. defended runs can land on the
            # same final answer (the model resisting the trick on its own),
            # so THIS is the part that actually differs between the two
            # buttons; burying it inside the full step trace below made that
            # easy to miss.
            seen_comment = None
            for step in result["steps"]:
                if step["action"] == "get_employee_record":
                    try:
                        seen_comment = json.loads(step["observation"]).get("manager_comment")
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    break
            if seen_comment:
                st.markdown("What the AI actually saw in E1's record:")
                st.code(seen_comment, language=None)

            st.markdown(f"Answer: {result['answer'] or '_(none)_'}")
            if result["guardrail_passed"]:
                st.success(f"✅ Output guardrail: {result['guardrail_reason']}")
            else:
                st.error(f"❌ Output guardrail: {result['guardrail_reason']} — this got through")
            with st.expander("Full step-by-step trace"):
                render_step_trace(result["steps"])
            st.divider()
