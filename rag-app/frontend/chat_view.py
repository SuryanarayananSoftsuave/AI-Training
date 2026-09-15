from __future__ import annotations

import streamlit as st

from utils.api_client import BackendClient

_VERDICT_STYLE = {
    "grounded": ("🟢", "Grounded"),
    "partially_grounded": ("🟡", "Partially grounded"),
    "hallucinated": ("🔴", "Hallucinated"),
    "no_answer": ("⚪", "No answer"),
    "out_of_scope": ("🚫", "Out of scope"),
    "judge_unavailable": ("⚠️", "Judge unavailable"),
}


def render_history(messages: list[dict]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_assistant_extras(message)


def _render_assistant_extras(message: dict) -> None:
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
        debug = message.get("retrieval_debug") or {}
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


def submit_query(client: BackendClient, query: str, settings: dict) -> dict:
    """Streams the answer text into the current `st.chat_message` via
    `st.write_stream`, then renders citations/judgment/debug once the
    stream's one "final" event arrives. Returns the assistant message dict
    to append to session state.

    The no-candidates and off-topic-gate-#1 short-circuits never stream any
    "delta" chunks (chat_service.py returns straight to a single "final"
    event for those), so `st.write_stream` sees an empty iterable and
    displays nothing -- the fallback below renders that final event's own
    `answer` text instead in that case.
    """
    final_payload: dict = {}

    def _text_deltas():
        for event_type, data in client.stream_chat(
            query=query,
            use_keyword_search=settings["use_keyword_search"],
            use_query_expansion=settings["use_query_expansion"],
            use_mmr=settings["use_mmr"],
            generator_provider=settings["generator_provider"],
            judge_provider=settings["judge_provider"],
            generator_temperature=settings["generator_temperature"],
            judge_temperature=settings["judge_temperature"],
            top_k=settings["top_k"],
            doc_ids=settings["doc_filter"] or None,
        ):
            if event_type == "delta":
                yield data
            else:
                final_payload.update(data)

    answer_text = st.write_stream(_text_deltas())
    if not answer_text:
        answer_text = final_payload.get("answer", "")
        if answer_text:
            st.markdown(answer_text)

    message = {
        "role": "assistant",
        "content": answer_text,
        "citations": final_payload.get("citations", []),
        "judgment": final_payload.get("judgment"),
        "retrieval_debug": final_payload.get("retrieval_debug"),
    }
    _render_assistant_extras(message)
    return message
