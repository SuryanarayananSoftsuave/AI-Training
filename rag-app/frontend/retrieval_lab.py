from __future__ import annotations

import streamlit as st

from utils.api_client import BackendClient

_SAMPLE_CHUNKING_TEXT = """# Leave Policy

## Section 2: Casual Leave Entitlement

Employees are entitled to 12 days of casual leave per calendar year, credited at the start of
January. Unused casual leave does not carry over to the next year and lapses on December 31.

## Section 3: Sick Leave

Employees may take up to 10 days of paid sick leave per year. A medical certificate is required
for any absence longer than 2 consecutive days. Sick leave that is not used in a given year
carries over up to a maximum of 5 days into the following year.

## Section 4: Bereavement Leave

Up to 5 days of paid bereavement leave are available following the death of an immediate family
member. Requests should be submitted to HR as soon as practical; documentation is not required.
"""


def render_retrieval_lab_panel(client: BackendClient) -> None:
    """Week 3/4 showcase: one question, 4 retrieval variants side by side --
    semantic vs hybrid search, each with and without MMR re-selection. No
    generation or judging happens here, so this is free to click repeatedly
    while exploring the leave-policy-2023-vs-2024 collision or any other
    retrieval question.
    """
    st.subheader("🔬 Retrieval Lab")
    st.caption(
        "Retrieval-only comparison, no LLM call: see exactly which chunks each search mode "
        "surfaces, and whether MMR swaps out a near-duplicate for something more diverse."
    )
    query = st.text_input(
        "Question", value="What was the casual leave entitlement under the 2023 policy specifically?",
        key="retrieval_lab_query",
    )
    top_k = st.slider("top-k per variant", min_value=1, max_value=10, value=5, key="retrieval_lab_top_k")

    if st.button("Compare", key="retrieval_lab_compare_btn"):
        with st.spinner("Retrieving + reranking 4 variants..."):
            try:
                result = client.compare_retrieval(query, top_k)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
        st.session_state["retrieval_lab_result"] = result

    result = st.session_state.get("retrieval_lab_result")
    if result:
        variants = result["variants"]  # order: semantic (no MMR), semantic+MMR, hybrid (no MMR), hybrid+MMR

        def _render_variant(col, variant: dict) -> None:
            with col:
                st.markdown(f"**{variant['label']}**")
                if not variant["chunks"]:
                    st.caption("(no candidates)")
                for c in variant["chunks"]:
                    with st.container(border=True):
                        st.caption(f"{c['filename']} p.{c.get('page_number')} · score {c['rerank_score']:.2f}")
                        if c.get("section_heading"):
                            st.caption(f"§ {c['section_heading']}")
                        st.caption(c["snippet"])

        # A genuine 2x2 grid -- semantic row, hybrid row -- rather than
        # one row of 4, so "no MMR" and "+ MMR" line up as columns
        # within each search mode's own row.
        for row_start in (0, 2):
            row_cols = st.columns(2)
            for col, variant in zip(row_cols, variants[row_start:row_start + 2]):
                _render_variant(col, variant)

    st.divider()
    _render_hit_rate_section(client)


def _render_hit_rate_section(client: BackendClient) -> None:
    st.markdown("**Week 4 — Hit-rate@3 (baseline vs. MMR)**")
    try:
        data = client.get_hit_rate_results()
    except Exception as exc:
        st.error(f"Could not load hit-rate results: {exc}")
        return

    if not data:
        st.info(
            "Not yet run. With Qdrant up and the HR policy PDFs indexed, from `backend/` run:\n\n"
            "`python scripts/run_hit_rate_eval.py`"
        )
        return

    col_a, col_b = st.columns(2)
    col_a.metric("Baseline hit-rate@3", f"{data['baseline_hit_rate']:.0%}", f"{data['baseline_hits']}/{data['n_questions']}")
    col_b.metric("With MMR hit-rate@3", f"{data['mmr_hit_rate']:.0%}", f"{data['mmr_hits']}/{data['n_questions']}")
    st.bar_chart({"hit-rate@3": {"baseline": data["baseline_hit_rate"], "with MMR": data["mmr_hit_rate"]}})
    if data["fixed"]:
        st.caption(f"Fixed by MMR: {data['fixed']}")
    if data["regressed"]:
        st.caption(f"⚠️ Regressed by MMR: {data['regressed']}")
    if data["still_missed"]:
        st.caption(f"Still missed: {data['still_missed']}")


def render_chunking_lab_panel(client: BackendClient) -> None:
    """Week 3 showcase: chunk the same text at two different (size, overlap)
    settings side by side -- a live answer to the still-open mentor-check
    item "did they try more than one chunk size and notice the difference?".
    No Qdrant/LLM call at all -- works even with Docker down.
    """
    st.subheader("✂️ Chunking Lab")
    st.caption(
        "Chunks the SAME text twice, at two different (size, overlap) settings, so you can see "
        "the actual difference instead of trusting the default (500/75 tokens) blindly."
    )
    text = st.text_area("Sample text to chunk", value=_SAMPLE_CHUNKING_TEXT, height=200, key="chunking_lab_text")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Setting A**")
        size_a = st.slider("Chunk size (tokens)", 20, 2000, 500, key="chunking_lab_size_a")
        # Capped strictly below size_a -- the splitter itself raises a
        # hard error for overlap >= chunk size, so the slider shouldn't
        # be able to reach an invalid combination in the first place.
        overlap_a = st.slider("Overlap (tokens)", 0, size_a - 1, min(75, size_a - 1), key="chunking_lab_overlap_a")
    with col2:
        st.markdown("**Setting B**")
        size_b = st.slider("Chunk size (tokens)", 20, 2000, 100, key="chunking_lab_size_b")
        overlap_b = st.slider("Overlap (tokens)", 0, size_b - 1, min(20, size_b - 1), key="chunking_lab_overlap_b")

    if st.button("Compare chunking", key="chunking_lab_compare_btn"):
        try:
            result = client.compare_chunking(text, size_a, overlap_a, size_b, overlap_b)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return
        st.session_state["chunking_lab_result"] = result

    result = st.session_state.get("chunking_lab_result")
    if result:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Setting A -> {len(result['chunks_a'])} chunk(s)**")
            for c in result["chunks_a"]:
                with st.container(border=True):
                    st.caption(f"{c['content_type']} · {c['token_count']} tokens · § {c.get('section_heading') or '(none)'}")
                    st.text(c["text"][:300])
        with col_b:
            st.markdown(f"**Setting B -> {len(result['chunks_b'])} chunk(s)**")
            for c in result["chunks_b"]:
                with st.container(border=True):
                    st.caption(f"{c['content_type']} · {c['token_count']} tokens · § {c.get('section_heading') or '(none)'}")
                    st.text(c["text"][:300])
