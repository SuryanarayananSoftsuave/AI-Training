from __future__ import annotations

import threading

import streamlit as st

from chat_view import _render_assistant_extras
from utils.api_client import BackendClient

# Ported from the original scratchpad `generate_traces.sh` -- a real, varied
# set of HR questions (leave-policy version-collision trap, out-of-
# jurisdiction questions that should hit the off-topic gate, questions
# spanning every policy doc) so a run produces genuine, diverse traces via
# chat_service.py's existing trace-capture instrumentation -- nothing here
# is fabricated, every answer is a real live call. Every question runs with
# whatever retrieval/model settings are currently set in the sidebar -- same
# as a normal chat message -- rather than a separate hardcoded config.
_BATCH_QUESTIONS: list[str] = [
    "How many casual leave days do I get?",
    "What is the current sick leave entitlement?",
    "How many days of earned leave carry over to the next year?",
    "What was the casual leave entitlement under the 2023 policy specifically?",
    "Is there a bereavement leave policy and how many days does it cover?",
    "What changed between the 2023 and 2024 leave policies?",
    "How many remote work days per week are employees allowed?",
    "What are the core hours for remote workers?",
    "What is the home office stipend amount and when is it paid?",
    "Can I work remotely five days a week permanently?",
    "How many days do I have to submit an expense claim?",
    "What's the approval threshold for expenses above $500?",
    "Can I expense alcohol at a client dinner?",
    "What happens if I lose a receipt for a small purchase?",
    "What value of gift needs to be disclosed under the code of conduct?",
    "How do I report a code of conduct violation confidentially?",
    "What happens if someone retaliates against a whistleblower?",
    "What is the monthly premium for employee-only health insurance?",
    "When is the open enrollment period for benefits?",
    "Can I add a dependent outside open enrollment?",
    "How often are performance reviews conducted?",
    "What rating is needed for promotion consideration?",
    "Does extended leave affect my performance review timing?",
    "What is the notice period for resignation in India?",
    "What is the notice period for resignation in the United States?",
    "What is the notice period for resignation in Germany?",
    "What is the notice period for resignation in France?",
    "What is the capital of Australia?",
    "Tell me about the plot of the movie Inception.",
]


def _run_one(client: BackendClient, query: str, settings: dict, state: dict) -> dict:
    """One real /chat call using the exact same settings a normal chat
    message would use (provider, temperature, top_k, doc filter, toggles --
    whatever was configured in the sidebar when "Run batch" was clicked).
    Checks `state["stop"]` between streamed chunks (not just between
    questions) so Stop cuts off a slow in-flight answer within about one
    chunk, instead of waiting out the whole remaining pipeline for the
    question already in progress.
    """
    answer = ""
    final_payload: dict = {}
    stopped_mid_stream = False
    try:
        stream = client.stream_chat(
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
        )
        for event_type, data in stream:
            if state["stop"]:
                stopped_mid_stream = True
                stream.close()
                break
            if event_type == "delta":
                answer += data
            else:
                final_payload = data
        if not answer:
            answer = final_payload.get("answer", "")
        if stopped_mid_stream:
            answer = (answer or "") + "\n\n_(stopped)_"
        return {
            "query": query,
            "content": answer,
            "citations": final_payload.get("citations", []),
            "judgment": final_payload.get("judgment"),
            "retrieval_debug": final_payload.get("retrieval_debug"),
            "error": None,
        }
    except Exception as exc:
        return {"query": query, "content": None, "citations": [], "judgment": None, "retrieval_debug": None, "error": str(exc)}


def _run_batch_worker(client: BackendClient, queries: list[str], settings: dict, state: dict) -> None:
    for query in queries:
        if state["stop"]:
            break
        state["results"].append(_run_one(client, query, settings, state))
    state["running"] = False


def _get_state() -> dict:
    if "batch_state" not in st.session_state:
        st.session_state.batch_state = {"running": False, "stop": False, "results": []}
    return st.session_state.batch_state


@st.fragment(run_every=1)
def _live_batch_panel(client: BackendClient, settings: dict) -> None:
    """A fragment auto-refreshes on its own timer without rerunning the rest
    of the page, so this panel updates live -- and Stop responds immediately
    -- while the main chat above stays fully usable. The background thread
    (started below) does the actual HTTP calls; this function only ever
    reads shared state and renders, never blocks.
    """
    state = _get_state()
    total = len(_BATCH_QUESTIONS)
    done = len(state["results"])

    if state["running"]:
        st.progress(done / total, text=f"Running... {done}/{total}")
        if st.button("⏹ Stop", key="stop_trace_batch", type="primary"):
            state["stop"] = True
    else:
        if st.button("▶ Run batch" if done == 0 else "▶ Run again", key="run_trace_batch"):
            state["results"] = []
            state["stop"] = False
            state["running"] = True
            # Snapshot the sidebar settings at click time -- same semantics
            # as a normal chat message using whatever's currently selected;
            # changing the sidebar mid-run doesn't retroactively affect a
            # batch already in flight.
            threading.Thread(
                target=_run_batch_worker, args=(client, _BATCH_QUESTIONS, dict(settings), state), daemon=True
            ).start()
            st.rerun(scope="fragment")
        elif done:
            note = " (stopped early)" if done < total else ""
            st.caption(f"Last run: {done}/{total} completed{note}")

    for result in state["results"]:
        with st.chat_message("user"):
            st.markdown(result["query"])
        with st.chat_message("assistant"):
            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown(result["content"] or "_(no answer)_")
                _render_assistant_extras(result)


def render_trace_batch_panel(client: BackendClient, settings: dict) -> None:
    with st.expander(f"🧪 Batch trace generator ({len(_BATCH_QUESTIONS)} questions)", expanded=False):
        st.caption(
            "Runs a curated set of real HR questions through the live chat pipeline, using "
            "the retrieval/model settings currently set in the sidebar for every question -- "
            "same as a normal chat message. Each produces a genuine trace in "
            "`data/traces/traces.jsonl`. Runs in the background, so the chat above stays "
            "usable while it runs."
        )
        _live_batch_panel(client, settings)
