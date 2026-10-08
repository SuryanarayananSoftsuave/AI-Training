from __future__ import annotations

import os

import streamlit as st

from agent_view import render_agent_race_panel, render_conversation_panel, render_dispatcher_panel, render_results_showcase_panel
from chat_view import render_history, submit_query
from eval_metrics import render_week6_eval_panel
from guardrail_playground import render_guardrail_playground_panel
from label_panel import render_week6_label_panel
from mcp_explorer import (
    render_mcp_ask_panel,
    render_mcp_discovery_panel,
    render_mcp_error_panel,
    render_mcp_risk_note_panel,
    render_mcp_wire_capture_panel,
)
from retrieval_lab import render_chunking_lab_panel, render_retrieval_lab_panel
from sidebar import render_sidebar
from trace_batch import render_trace_batch_panel
from trace_explorer import render_trace_explorer_panel
from trajectory_explorer import render_injection_playground_panel, render_trajectory_eval_panel
from utils.api_client import BackendClient

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋", layout="wide")

if "client" not in st.session_state:
    st.session_state.client = BackendClient(BACKEND_URL)
if "messages" not in st.session_state:
    st.session_state.messages = []

client: BackendClient = st.session_state.client

settings = render_sidebar(client, BACKEND_URL)

st.title("📋 HR Policy Assistant")

tab_ask, tab_log = st.tabs(["💬  Ask", "🛠️  Build Log"])

with tab_ask:
    render_history(st.session_state.messages)

    if query := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                message = submit_query(client, query, settings)
                st.session_state.messages.append(message)
            except Exception as exc:
                st.error(f"Chat failed: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"⚠ Chat failed: {exc}"})
        st.rerun()

with tab_log:
    st.caption(
        "The engineering work behind this assistant, week by week -- every result here is real "
        "and computed live when you click, not a stored report."
    )

    week2, week34, week5, week6, week7, week8, week9 = st.tabs(
        ["Week 2", "Week 3–4", "Week 5", "Week 6", "Week 7", "Week 8", "Week 9"]
    )

    with week2:
        render_guardrail_playground_panel(client)

    with week34:
        sub_chunk, sub_retrieval = st.tabs(["Chunking Lab", "Retrieval Lab"])
        with sub_chunk:
            render_chunking_lab_panel(client)
        with sub_retrieval:
            render_retrieval_lab_panel(client)

    with week5:
        sub_batch, sub_explorer = st.tabs(["Batch Trace Generator", "Trace Explorer"])
        with sub_batch:
            render_trace_batch_panel(client, settings)
        with sub_explorer:
            render_trace_explorer_panel(client)

    with week6:
        sub_eval, sub_label = st.tabs(["Automated Eval", "Blind Labeling"])
        with sub_eval:
            render_week6_eval_panel(client)
        with sub_label:
            render_week6_label_panel(client)

    with week7:
        sub_results, sub_race, sub_dispatch, sub_conv = st.tabs(
            ["Results", "Agent vs Workflow", "Hybrid Dispatcher", "Conversation Memory"]
        )
        with sub_results:
            render_results_showcase_panel(client)
        with sub_race:
            render_agent_race_panel(client)
        with sub_dispatch:
            render_dispatcher_panel(client)
        with sub_conv:
            render_conversation_panel(client)

    with week8:
        sub_traj, sub_inject = st.tabs(["Trajectory Eval", "Injection Playground"])
        with sub_traj:
            render_trajectory_eval_panel(client)
        with sub_inject:
            render_injection_playground_panel(client)

    with week9:
        sub_discover, sub_wire, sub_error, sub_ask, sub_risk = st.tabs(
            ["Discovery", "Wire Capture", "Recoverable Errors", "Ask via MCP", "Risk Note"]
        )
        with sub_discover:
            render_mcp_discovery_panel(client)
        with sub_wire:
            render_mcp_wire_capture_panel(client)
        with sub_error:
            render_mcp_error_panel(client)
        with sub_ask:
            render_mcp_ask_panel(client)
        with sub_risk:
            render_mcp_risk_note_panel(client)
