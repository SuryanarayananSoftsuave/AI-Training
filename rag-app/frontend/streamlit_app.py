from __future__ import annotations

import os

import streamlit as st

from chat_view import render_history, submit_query
from sidebar import render_sidebar
from trace_batch import render_trace_batch_panel
from utils.api_client import BackendClient

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="HR Policy RAG", page_icon="📋", layout="wide")

if "client" not in st.session_state:
    st.session_state.client = BackendClient(BACKEND_URL)
if "messages" not in st.session_state:
    st.session_state.messages = []

client: BackendClient = st.session_state.client

settings = render_sidebar(client, BACKEND_URL)

st.title("📋 HR Policy RAG")

render_trace_batch_panel(client, settings)

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
