from __future__ import annotations

import streamlit as st

from utils.api_client import BackendClient


def render_week6_label_panel(client: BackendClient) -> None:
    """Week 6, step 3: blind-label each raw answer (from
    evals/week6_raw_answers.json, judge never having run on it) as
    grounded/not-grounded yourself. The judge's verdict is never fetched or
    shown here -- see app/api/evals.py's next-unlabeled route -- so the
    resulting evals/labels_25.json is genuinely blind, which is what the
    W6-Task-Set-C.md rubric's ordering-proof requirement (#3) needs.
    """
    st.subheader("🏷️ Blind Labeling")
    st.caption("Label each answer grounded/not-grounded yourself, before the judge ever sees it.")

    try:
        state = client.get_week6_labeling_state()
    except Exception as exc:
        st.error(f"Could not load labeling state: {exc}")
        return

    st.caption(f"Criterion: *{state['criterion']}*")
    # Clamped defensively -- the backend already filters labeled_count to
    # the current answer set, but a stale/corrupted labels_25.json
    # pushing this above 1.0 would otherwise crash Streamlit's st.progress.
    fraction = min(state["labeled_count"] / state["total"], 1.0) if state["total"] else 0
    st.progress(fraction, text=f"{state['labeled_count']}/{state['total']} labeled")

    if state["all_labeled"]:
        st.success("All cases labeled. Commit this now, before running the judge:")
        st.code(
            "cd rag-app/backend\n"
            "git add evals/labels_25.json\n"
            'git commit -m "Week 6: blind labels, predate judge run"\n'
            "python scripts/run_week6_eval.py",
            language="bash",
        )
        return

    try:
        case = client.get_week6_next_unlabeled()
    except Exception as exc:
        st.error(f"Could not load next case: {exc}")
        return

    if case["case_id"] is None:
        st.info("Nothing to label.")
        return

    st.markdown(f"**[{case['mode']}] `{case['case_id']}`**")
    st.markdown(f"**Q:** {case['question']}")
    st.markdown(f"**A:** {case['answer']}")
    if case["citations"]:
        st.caption(
            "Citations: "
            + "; ".join(f"[{c['marker']}] {c['filename']} p.{c.get('page_number')}" for c in case["citations"])
        )
    else:
        st.caption("Citations: (none)")

    st.caption("⚠️ The judge's verdict is intentionally never fetched here -- your label must not be influenced by it.")

    col_pass, col_fail = st.columns(2)
    if col_pass.button("✅ Grounded", key=f"label_pass_{case['case_id']}", use_container_width=True):
        try:
            client.submit_week6_label(case["case_id"], True)
        except Exception as exc:
            st.error(f"Could not save label (it may have already been labeled elsewhere): {exc}")
        else:
            st.rerun()
    if col_fail.button("❌ Not grounded", key=f"label_fail_{case['case_id']}", use_container_width=True):
        try:
            client.submit_week6_label(case["case_id"], False)
        except Exception as exc:
            st.error(f"Could not save label (it may have already been labeled elsewhere): {exc}")
        else:
            st.rerun()
