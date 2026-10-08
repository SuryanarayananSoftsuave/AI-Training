from __future__ import annotations

import httpx
import streamlit as st

from utils.api_client import BackendClient

_VERDICT_EXPLANATION = {
    "out_of_scope": "🚫 Off-topic/out-of-jurisdiction guardrail fired -- a fixed refusal was returned "
    "without ever calling the generator/judge a second time, or the generator cited zero sources "
    "and was overridden. Your code decided this, not the model.",
    "no_answer": "⚪ No-candidates guardrail fired -- retrieval found nothing, so a fixed "
    "\"I couldn't find anything\" response was returned without any generation call at all.",
    "judge_unavailable": "⚠️ The judge call itself failed (e.g. a transient provider error) and the "
    "pipeline degraded gracefully instead of crashing the whole request.",
    "grounded": "✅ Normal grounded answer -- no guardrail needed to intervene.",
    "partially_grounded": "🟡 Answer produced, partially grounded per the judge -- no hard guardrail fired.",
    "hallucinated": "🔴 Answer produced, judge flagged unsupported claims -- no hard guardrail fired.",
}

_PRESETS = {
    "Off-topic question": "What is the capital of Australia?",
    "Out-of-jurisdiction question": "What is the notice period for resignation in Germany?",
    "Normal in-scope question (baseline)": "How many casual leave days do I get?",
}


def _run_and_explain(client: BackendClient, query: str) -> None:
    final_payload: dict = {}
    answer = ""
    try:
        for event_type, data in client.stream_chat(
            query=query, use_keyword_search=True, use_query_expansion=False, use_mmr=False,
            generator_provider="gemini", judge_provider="gemini",
            generator_temperature=0.2, judge_temperature=0.0, top_k=6, doc_ids=None,
        ):
            if event_type == "delta":
                answer += data
            else:
                final_payload = data
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 422:
            st.warning(
                "✅ Guardrail: the request never reached the LLM -- rejected by Pydantic validation "
                "before any pipeline code ran (HTTP 422)."
            )
            st.code(exc.response.text, language="json")
        else:
            # A genuine backend/provider failure (500, 503, ...) is NOT a
            # guardrail catching anything -- it's the opposite outcome, and
            # must never be shown with the same green "guardrail worked" framing.
            st.error(f"❌ Backend error (HTTP {exc.response.status_code}) -- not a guardrail, something went wrong:")
            st.code(exc.response.text, language="json")
        return
    except Exception as exc:
        st.error(f"Request failed unexpectedly: {exc}")
        return

    verdict = (final_payload.get("judgment") or {}).get("verdict")
    st.markdown(f"**Answer:** {answer or final_payload.get('answer', '') or '_(none)_'}")
    st.info(_VERDICT_EXPLANATION.get(verdict, f"Verdict: {verdict}"))


def render_guardrail_playground_panel(client: BackendClient) -> None:
    """Week 2 showcase: fires deliberately adversarial inputs at the real
    `/chat` endpoint and names which guardrail actually caught each one --
    making "your code, not the AI, decides" a clickable demo instead of
    something only encountered by accident. No new backend code: this reuses
    the existing off-topic gates, no-candidates gate, and Pydantic request
    validation exactly as `/chat` already enforces them.
    """
    st.subheader("🛡️ Guardrail Playground")
    st.caption(
        "Sends a real request to /chat and explains which guardrail (if any) intervened -- "
        "the off-topic gates, the no-candidates gate, or Pydantic validation at the request boundary."
    )

    preset = st.selectbox("Preset", list(_PRESETS.keys()) + ["Oversized query (>2000 chars)", "Custom..."])
    if preset == "Custom...":
        query = st.text_input("Your question", key="guardrail_custom_query")
    elif preset == "Oversized query (>2000 chars)":
        query = "What is the leave policy? " * 120  # comfortably over ChatRequest's max_length=2000
        st.caption(f"({len(query)} characters -- ChatRequest.query caps at 2000)")
    else:
        query = _PRESETS[preset]

    if st.button("Test guardrail", key="guardrail_test_btn") and query:
        with st.spinner("Running..."):
            _run_and_explain(client, query)
