from __future__ import annotations

import os

import streamlit as st

from utils.api_client import BackendClient

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support RAG", page_icon="🎧", layout="wide")

if "client" not in st.session_state:
    st.session_state.client = BackendClient(BACKEND_URL)
if "messages" not in st.session_state:
    st.session_state.messages = []

client: BackendClient = st.session_state.client

_VERDICT_STYLE = {
    "grounded": ("🟢", "Grounded"),
    "partially_grounded": ("🟡", "Partially grounded"),
    "hallucinated": ("🔴", "Hallucinated"),
    "no_answer": ("⚪", "No answer"),
    "out_of_scope": ("🚫", "Out of scope"),
}
_STATUS_BADGE = {"indexed": "🟢", "processing": "🟡", "pending": "⚪", "failed": "🔴"}

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
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
        st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")

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

st.title("🎧 Customer Support RAG")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations")
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for c in citations:
                    st.markdown(f"**[{c['marker']}] {c['filename']}**, page {c['page_number']} — score {c['rerank_score']:.2f}")
                    st.caption(c["snippet"])
        judgment = message.get("judgment")
        if judgment:
            icon, label = _VERDICT_STYLE.get(judgment["verdict"], ("⚪", judgment["verdict"]))
            st.caption(f"{icon} **{label}**")
            debug = message.get("retrieval_debug", {})
            st.caption(
                f"mode: {debug.get('mode')} · candidates: {debug.get('candidates_after_fusion')} "
                f"→ reranked: {debug.get('reranked_returned')} · "
                f"generator: {debug.get('generator_provider')}@{debug.get('generator_temperature')} · "
                f"judge: {debug.get('judge_provider')}@{debug.get('judge_temperature')}"
            )
            variants = debug.get("query_variants")
            if variants:
                st.markdown("**Also searched with:**")
                for v in variants:
                    st.markdown(f"- {v}")

if query := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = client.chat(
                    query=query,
                    use_keyword_search=st.session_state.use_keyword_search,
                    use_query_expansion=st.session_state.use_query_expansion,
                    generator_provider=generator_provider,
                    judge_provider=judge_provider,
                    generator_temperature=generator_temperature,
                    judge_temperature=judge_temperature,
                    top_k=top_k,
                    doc_ids=doc_filter or None,
                )
                st.markdown(result["answer"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result["citations"],
                        "judgment": result["judgment"],
                        "retrieval_debug": result["retrieval_debug"],
                    }
                )
            except Exception as exc:
                st.error(f"Chat failed: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"⚠ Chat failed: {exc}"})
    st.rerun()
