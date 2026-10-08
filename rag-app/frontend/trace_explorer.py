from __future__ import annotations

import streamlit as st

from utils.api_client import BackendClient

_SEVERITY_OPTIONS = ["Legal/compliance exposure", "Annoys the employee only"]


def render_trace_explorer_panel(client: BackendClient) -> None:
    """Week 5 showcase: sample -> open-code -> cluster into modes -> dated
    prediction -> export, guided inside the app instead of a CLI script plus
    hand-edited notes.md/taxonomy.md files. Every step calls the real
    backend (app/api/analysis.py), which in turn reuses the real
    TraceStore.sample_random the CLI script already uses -- nothing here is
    a second, parallel implementation of the sampling logic.
    """
    st.subheader("🔎 Trace Explorer")
    st.caption(
        "Sample real traces, open-code each one, cluster into named failure modes, write a dated "
        "prediction, then export -- guided inside the app instead of a CLI script plus hand-edited files."
    )
    seed = st.number_input("Seed", value=42, step=1, key="week5_seed")

    # The sentinel below stops a FAILED fetch from refiring on every
    # unrelated rerun of this large multi-panel page (any click/toggle
    # anywhere reruns the whole script) -- only an explicit button
    # click retries after a failure; success is still cached as before.
    need_fetch = "week5_sample" not in st.session_state and not st.session_state.get("week5_sample_attempted")
    if st.button("Sample 20 traces", key="week5_sample_btn") or need_fetch:
        st.session_state["week5_sample_attempted"] = True
        try:
            st.session_state["week5_sample"] = client.get_week5_sample(seed=int(seed), n=20)
        except Exception as exc:
            st.info(f"No sample yet: {exc}")
            return

    sample = st.session_state.get("week5_sample")
    if sample is None:
        st.info("Click 'Sample 20 traces' to get started.")
        return
    st.caption(f"Seed {sample['seed']} · sampled at {sample['sampled_at']}")
    trace_ids = [t["trace_id"] for t in sample["traces"]]

    st.markdown("**Step 1 — Open-code each trace**")
    idx = st.selectbox(
        "Trace",
        options=list(range(len(trace_ids))),
        format_func=lambda i: f"{i + 1}. {sample['traces'][i]['question'][:60]}",
        key="week5_trace_idx",
    )
    trace_id = trace_ids[idx]
    # Cached per trace_id -- this page reruns on almost any interaction
    # anywhere on it, and re-fetching the same trace's detail on every
    # unrelated rerun is pure waste; a genuinely different selection
    # still fetches fresh (and after "Save sentence" below, the
    # textarea's own widget state already reflects the saved text, so
    # this cache never shows a stale sentence for the same trace).
    if st.session_state.get("week5_detail_trace_id") != trace_id:
        try:
            st.session_state["week5_detail"] = client.get_week5_trace_detail(trace_id)
            st.session_state["week5_detail_trace_id"] = trace_id
        except Exception as exc:
            st.error(f"Could not load trace: {exc}")
            return
    detail = st.session_state["week5_detail"]

    st.markdown(f"**Q:** {detail['question']}")
    st.markdown(f"**A:** {detail['answer']}")
    if detail["ranked_candidates"]:
        st.caption(
            "Retrieved: "
            + "; ".join(
                f"{c['filename']} p.{c.get('page_number')} (score {c['rerank_score']:.2f}"
                f"{', cited' if c['cited'] else ''})"
                for c in detail["ranked_candidates"]
            )
        )
    if detail["judgment"]:
        st.caption(f"Judgment: {detail['judgment']['verdict']} (confidence {detail['judgment']['confidence']})")
    if detail["error"]:
        st.caption(f"⚠️ Error: {detail['error']}")

    sentence = st.text_area(
        "One honest sentence — what you SEE, not a category or a fix",
        value=detail.get("sentence") or "",
        key=f"week5_sentence_{trace_id}",
    )
    if st.button("Save sentence", key=f"week5_save_sentence_{trace_id}"):
        result = client.set_week5_sentence(trace_id, sentence)
        st.success(f"Coded {result['coded_count']}/{result['total']}")

    st.divider()
    st.markdown("**Step 2 — Cluster into named modes**")
    _render_mode_builder(client, trace_ids)

    st.divider()
    st.markdown("**Step 3 — Dated prediction + benchmark note**")
    _render_prediction_form(client)

    st.divider()
    if st.button("Export notes.md / taxonomy.md / prediction.txt", key="week5_export_btn"):
        try:
            exported = client.export_week5()
        except Exception as exc:
            st.error(f"Export failed: {exc}")
            return
        st.success(f"Written to `{exported['written_to']}`")
        for filename, content in exported["files"].items():
            st.markdown(f"**{filename}**")
            st.code(content, language="markdown" if filename.endswith(".md") else None)
        st.code(
            "cd rag-app/backend\n"
            "git add analysis/\n"
            'git commit -m "Week 5: taxonomy, notes, dated prediction"',
            language="bash",
        )


def _render_mode_builder(client: BackendClient, trace_ids: list[str]) -> None:
    if "week5_modes" not in st.session_state:
        st.session_state["week5_modes"] = []

    with st.form("week5_add_mode_form", clear_on_submit=True):
        name = st.text_input("Mode name (stranger-legible, e.g. 'cites the superseded 2023 leave policy')")
        severity = st.selectbox("Severity", _SEVERITY_OPTIONS)
        members = st.multiselect("Which traces belong to this mode?", options=trace_ids)
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Add mode") and name and members:
            st.session_state["week5_modes"].append({"name": name, "trace_ids": members, "severity": severity, "note": note})

    modes = st.session_state["week5_modes"]
    if modes:
        n = len(trace_ids) or 1
        rows = [
            {"mode": m["name"], "count": len(m["trace_ids"]), "%": round(100 * len(m["trace_ids"]) / n, 1), "severity": m["severity"]}
            for m in modes
        ]
        st.table(rows)
        if len(modes) < 4:
            st.caption("⚠️ Fewer than 4 categories found — aim for 4-7 for a well-rounded taxonomy.")
        elif len(modes) > 7:
            st.caption("⚠️ More than 7 categories found — consider merging similar ones; aim for 4-7.")
        if st.button("Save modes", key="week5_save_modes_btn"):
            client.set_week5_modes(modes)
            st.success("Modes saved.")


def _render_prediction_form(client: BackendClient) -> None:
    mode = st.text_input("Which mode will you attack next?", key="week5_pred_mode")
    change = st.text_input("What specific change will you make?", key="week5_pred_change")
    delta = st.text_input("Exact expected delta (e.g. 'drops from 30% to under 10%')", key="week5_pred_delta")
    if st.button("Save prediction (do this BEFORE making the change)", key="week5_save_pred_btn"):
        if mode and change and delta:
            result = client.set_week5_prediction(mode, change, delta)
            st.success(f"Prediction saved at {result['written_at']}")
        else:
            st.warning("Fill in all three fields.")

    note = st.text_area("3 sentences: why a public benchmark would've missed your top modes", key="week5_benchmark_note")
    if st.button("Save benchmark note", key="week5_save_benchmark_btn"):
        client.set_week5_benchmark_note(note)
        st.success("Saved.")
