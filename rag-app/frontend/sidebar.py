from __future__ import annotations

import streamlit as st

from utils.api_client import BackendClient

_STATUS_BADGE = {"indexed": "🟢", "processing": "🟡", "pending": "⚪", "failed": "🔴"}


def render_sidebar(client: BackendClient, backend_url: str) -> dict:
    """Renders the upload panel, document list, and retrieval/model settings
    controls. Returns the collected settings dict the chat loop needs to
    build each request.
    """
    with st.sidebar:
        header_col, refresh_col = st.columns([4, 1])
        header_col.header("Documents")
        # `list_documents()` below is already called fresh on every script
        # run -- Streamlit reruns the whole script on any widget click, so
        # this button needs no special handling, it just gives the user a
        # way to trigger that rerun without uploading/deleting anything,
        # e.g. to poll a `processing` doc until it flips to `indexed`.
        if refresh_col.button("🔄", help="Refresh document statuses", use_container_width=True):
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "pptx", "md", "markdown", "txt", "xlsx", "csv", "rtf", "html", "htm"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Upload", use_container_width=True):
            for uploaded in uploaded_files:
                try:
                    result = client.upload_document(uploaded.name, uploaded.getvalue())
                    (st.info if result.get("duplicate_of") else st.success)(f"{uploaded.name}: {result['message']}")
                except Exception as exc:
                    st.error(f"{uploaded.name}: upload failed — {exc}")
            st.rerun()

        st.divider()

        try:
            docs = client.list_documents()
        except Exception as exc:
            docs = []
            st.error(f"Could not reach backend at {backend_url}: {exc}")

        if docs:
            for doc in docs:
                badge = _STATUS_BADGE.get(doc["status"], "⚪")
                with st.container(border=True):
                    st.markdown(f"**{doc['original_filename']}** {badge} `{doc['status']}`")
                    st.caption(
                        f"{doc.get('num_pages') or '–'} pages · {doc.get('num_chunks') or '–'} chunks · "
                        f"hash `{doc['file_hash'][:10]}…`"
                    )
                    if doc.get("error"):
                        st.caption(f"⚠ {doc['error']}")
                    if st.button("Delete", key=f"del_{doc['doc_id']}"):
                        client.delete_document(doc["doc_id"])
                        st.rerun()
        else:
            st.caption("No documents uploaded yet.")

        st.divider()

        with st.expander("Retrieval settings", expanded=True):
            use_keyword_search = st.toggle(
                "Enable keyword search (hybrid)",
                key="use_keyword_search",
                help="Off: semantic-only dense search. On: dense + BM25 sparse search, fused with RRF.",
            )
            use_query_expansion = st.toggle(
                "Enable query expansion (multi-query)",
                key="use_query_expansion",
                help="Generates 3 alternate phrasings of your question, retrieves for all of them in "
                "parallel, and merges the results -- helps when your wording doesn't match the "
                "document's wording. Costs one extra Gemini call and more retrieval time per question.",
            )
            use_mmr = st.toggle(
                "Enable MMR (diversity re-selection)",
                key="use_mmr",
                help="Re-selects the final top-k from a larger reranked pool, trading a little "
                "relevance for diversity -- helps when several near-duplicate chunks (e.g. an "
                "old vs. current policy version) crowd out a genuinely different, relevant one.",
            )
            top_k = st.slider("Answer built from top-k chunks", min_value=1, max_value=15, value=6)
            indexed_docs = [d for d in docs if d["status"] == "indexed"]
            doc_filter = st.multiselect(
                "Limit to specific documents",
                options=[d["doc_id"] for d in indexed_docs],
                format_func=lambda doc_id: next(d["original_filename"] for d in indexed_docs if d["doc_id"] == doc_id),
            )

        with st.expander("Model settings", expanded=True):
            generator_provider = st.selectbox(
                "Generator LLM",
                options=["gemini", "groq"],
                format_func=lambda p: p.capitalize(),
                help="The model that writes the answer.",
            )
            judge_provider = st.selectbox(
                "Judge LLM",
                options=["gemini", "groq"],
                format_func=lambda p: p.capitalize(),
                help="The model that fact-checks the answer against the retrieved context. Pick a "
                "different provider than the generator to keep the cross-provider hallucination check "
                "meaningful -- a model judging its own answer is a weaker check.",
            )
            generator_temperature = st.slider(
                "Generator temperature",
                min_value=0.0, max_value=2.0, value=0.2, step=0.1,
                help="Low = sticks closely to the retrieved context (recommended for factual answers). "
                "High = more creative/varied wording, more prone to drifting from the source text.",
            )
            judge_temperature = st.slider(
                "Judge temperature",
                min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                help="Low (0 = deterministic) is recommended -- a fact-checker that gives different "
                "verdicts on repeated runs of the same answer is a weaker check.",
            )

    return {
        "use_keyword_search": use_keyword_search,
        "use_query_expansion": use_query_expansion,
        "use_mmr": use_mmr,
        "top_k": top_k,
        "doc_filter": doc_filter,
        "generator_provider": generator_provider,
        "judge_provider": judge_provider,
        "generator_temperature": generator_temperature,
        "judge_temperature": judge_temperature,
    }
