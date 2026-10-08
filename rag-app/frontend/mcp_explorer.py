"""Week 9 (W9-Task-Set-C.md) showcase: five panels, one per graded
requirement, all driven by real backend calls (app/api/mcp.py) -- every
number/name/frame shown here is live, never a stored screenshot.
"""
from __future__ import annotations

import json

import streamlit as st

from agent_view import render_step_trace
from utils.api_client import BackendClient

_WIRE_CAPTURE_PRESETS: dict[str, tuple[str, str, dict]] = {
    "policy-search / search_handbook": ("policy-search", "search_handbook", {"query": "leave policy"}),
    "hris / get_grade_band": ("hris", "get_grade_band", {"employee_id": "E3"}),
    "hris / get_accrued_leave_balance": ("hris", "get_accrued_leave_balance", {"employee_id": "E3"}),
}

_ASK_EXAMPLES = [
    "What grade band is employee E3 in?",
    "How many accrued leave days does employee E8 have?",
    "What does the leave policy say about casual leave?",
]


def render_mcp_discovery_panel(client: BackendClient) -> None:
    """Requirement #1 (base build) + #3 (15 pts): connect with 1 server,
    then add the HRIS server with nothing but a config swap -- the tool
    count and names shown below come straight from a live tools/list call,
    and the diff proofs underneath run `git diff --no-index` live too.
    """
    st.subheader("🔌 Discovery")
    st.caption(
        "Connect with the policy-search server alone, then add the HRIS server -- config only, "
        "zero lines changed in agents/mcp_agent.py. Every number below is a live tools/list result."
    )

    col1, col2 = st.columns(2)
    if col1.button("Connect (1 server)", key="mcp_connect_one"):
        with st.spinner("Connecting..."):
            st.session_state["mcp_tools_result"] = client.get_mcp_tools("one")
    if col2.button("Add server 2 (HRIS)", key="mcp_connect_two"):
        with st.spinner("Reconnecting with server 2..."):
            st.session_state["mcp_tools_result"] = client.get_mcp_tools("two")

    result = st.session_state.get("mcp_tools_result")
    if result:
        st.markdown(f"**Config: `{result['config_label']}`-server -- {result['count']} tool(s) discovered**")
        for tool in result["tools"]:
            with st.container(border=True):
                st.markdown(f"`{tool['name']}`")
                st.caption(tool["description"] or "_(no description)_")

    st.divider()
    st.markdown("**Proof: adding the second server required zero code changes**")
    if st.button("Check agent_diff.txt (should be empty)", key="mcp_agent_diff_btn"):
        with st.spinner("Running git diff --no-index..."):
            diff = client.get_mcp_agent_diff()
        if diff["is_empty"]:
            st.success("✅ agents/mcp_agent.py: 0 changed lines between server-1-only and server-1+2.")
        else:
            st.error("Unexpected: mcp_agent.py differs from its baseline.")
            st.code(diff["diff_text"], language="diff")

    st.markdown("**Config diff (what actually changed to add server 2)**")
    if st.button("Show config diff", key="mcp_config_diff_btn"):
        with st.spinner("Running git diff --no-index..."):
            diff = client.get_mcp_config_diff()
        st.code(diff["diff_text"] or "(no differences)", language="diff")


def render_mcp_wire_capture_panel(client: BackendClient) -> None:
    """Requirement #4 (25 pts): a real initialize -> tools/list -> tools/call
    capture against whichever tool is picked, rendered alongside the
    hand-written field-by-field annotations.
    """
    st.subheader("📡 Wire Capture")
    st.caption(
        "Runs a real JSON-RPC exchange against one tool and shows every raw frame, annotated -- "
        "the model itself never runs inside the server process; see the note below."
    )

    preset = st.selectbox("Tool to call", list(_WIRE_CAPTURE_PRESETS.keys()), key="mcp_wire_preset")
    if st.button("Capture", key="mcp_wire_capture_btn"):
        server, tool, args = _WIRE_CAPTURE_PRESETS[preset]
        with st.spinner("Capturing raw JSON-RPC frames..."):
            st.session_state["mcp_wire_result"] = client.mcp_wire_capture(server, tool, args)

    result = st.session_state.get("mcp_wire_result")
    if result:
        st.markdown(f"**{len(result['frames'])} frame(s) captured**")
        for i, entry in enumerate(result["frames"], 1):
            frame = entry["frame"]
            label = frame.get("method") or ("response" if "result" in frame else "notification")
            with st.expander(f"Frame {i}: {entry['direction']} -- `{label}`"):
                st.code(json.dumps(frame, indent=2), language="json")
        with st.expander("📖 Hand-written field annotations", expanded=True):
            st.markdown(result["annotations_markdown"])


def render_mcp_error_panel(client: BackendClient) -> None:
    """Requirement #5 (20 pts): the same failing search_handbook(as_of=...)
    call, run through the real agent, against the deliberately-worse
    "before" server and the real "after" server -- showing how the MODEL's
    answer changes, not just the raw error text.
    """
    st.subheader("🩹 Recoverable Errors")
    st.caption(
        "Same question, same failing tool call, two server versions -- the docstring and error "
        "message are the only difference, on OUR OWN policy-search server."
    )

    col1, col2 = st.columns(2)
    if col1.button("Run BEFORE", key="mcp_error_before_btn"):
        with st.spinner("Running (before)..."):
            st.session_state["mcp_error_before"] = client.mcp_error_demo("before")
    if col2.button("Run AFTER", key="mcp_error_after_btn"):
        with st.spinner("Running (after)..."):
            st.session_state["mcp_error_after"] = client.mcp_error_demo("after")

    col1, col2 = st.columns(2)
    for col, key, label in ((col1, "mcp_error_before", "Before"), (col2, "mcp_error_after", "After")):
        entry = st.session_state.get(key)
        with col:
            st.markdown(f"**{label}**")
            if entry:
                render_step_trace(entry["steps"])
                st.info(entry["answer"])
            else:
                st.caption("_(not run yet)_")


def render_mcp_ask_panel(client: BackendClient) -> None:
    """Ties it together: a real question, run end-to-end through
    agents/mcp_agent.py against both servers -- discover, pick a tool, call
    it, answer.
    """
    st.subheader("💬 Ask via MCP")
    st.caption("Runs a real question through the MCP-discovered agent (both servers connected).")

    preset = st.selectbox("Example", _ASK_EXAMPLES + ["Custom..."], key="mcp_ask_preset")
    question = st.text_input("Your question", key="mcp_ask_custom") if preset == "Custom..." else preset

    if st.button("Ask", key="mcp_ask_btn") and question:
        with st.spinner("Running..."):
            st.session_state["mcp_ask_result"] = client.ask_mcp(question)

    result = st.session_state.get("mcp_ask_result")
    if result:
        render_step_trace(result["steps"])
        st.success(result["answer"])


def render_mcp_risk_note_panel(client: BackendClient) -> None:
    """A short supply-chain risk write-up for the HRIS server -- who built
    it, what it can reach, what it logs, what a stolen credential could do,
    and a ship/don't call, since bolting on a third-party-style tool means
    trusting code that isn't ours.
    """
    st.subheader("⚠️ Risk Note")
    st.caption(
        "Connecting to a third-party-style tool means trusting code outside this app -- this note is a "
        "starting-point risk assessment, drafted for review rather than a final answer."
    )

    note = client.get_mcp_risk_note()
    if note.get("text"):
        st.markdown(note["text"])
    else:
        st.warning("No risk note found yet.")
