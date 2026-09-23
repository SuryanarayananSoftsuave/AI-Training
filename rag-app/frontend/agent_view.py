from __future__ import annotations

import json
import uuid

import streamlit as st

from utils.api_client import BackendClient

_SAMPLE_QUESTIONS = [
    "What is the notice period for employee E1 if they resign?",
    "What is the notice period for employee E2 if they resign?",
    "How many days of notice does employee E3 need to give?",
    "How many days of notice does employee E4 need to give?",
    "What notice period applies to employee E5?",
    "What notice period applies to employee E6?",
    "Employee E7 wants to resign -- what is their required notice period?",
    "Employee E8 wants to resign -- what is their required notice period?",
    "Employee E9 wants to resign -- what is their required notice period?",
    "Employee E10 wants to resign -- what is their required notice period?",
    "Compare the notice periods for employee E1 and employee E2 -- which one needs to give more notice?",
]


def render_agent_race_panel(client: BackendClient) -> None:
    """Week 7: pick one of the 10 race questions (or type your own), see
    the ReAct agent's live step-by-step tool-call trace side by side with
    the fixed workflow's single-shot answer -- same question, same tools,
    same model, two different control-flow strategies.
    """
    with st.expander("Week 7 - Agent vs Workflow", expanded=False):
        st.caption(
            "Same HR question, two implementations: a ReAct-style tool-calling agent that decides its "
            "own next step each lap, vs a fixed workflow with hard-coded steps. Both use the same 3 "
            "tools and the same model -- only the control flow differs."
        )
        question = st.selectbox("Question", _SAMPLE_QUESTIONS, index=0)
        custom = st.text_input("...or type your own (must name an employee, e.g. 'E3')", value="")
        final_question = custom.strip() or question

        if st.button("Ask both", key="ask_agent_btn"):
            with st.spinner("Running agent + workflow..."):
                try:
                    result = client.ask_agent(final_question)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    return
            st.session_state["agent_race_result"] = result

        result = st.session_state.get("agent_race_result")
        if not result:
            return

        col_agent, col_workflow = st.columns(2)

        with col_agent:
            st.subheader("🔁 Agent (ReAct loop)")
            st.markdown(f"**Answer:** {result['agent_answer'] or '_(none -- budget fired before finishing)_'}")
            st.caption(
                f"{result['agent_iterations']} lap(s) · {result['agent_tokens']} tokens · "
                f"{result['agent_latency_s']:.2f}s · terminated: {result['agent_terminated_reason'] or 'normally'}"
            )
            for i, step in enumerate(result["agent_steps"], 1):
                with st.container(border=True):
                    st.markdown(f"**Step {i}: `{step['action']}`**")
                    st.caption(f"Thought: {step['thought']}")
                    st.code(json.dumps(step["action_input"]), language="json")
                    st.caption(f"Observation: {step['observation']}")

        with col_workflow:
            st.subheader("📋 Workflow (fixed steps)")
            if result["workflow_error"]:
                st.error(result["workflow_error"])
            else:
                st.markdown(f"**Answer:** {result['workflow_answer']}")
            st.caption(f"{result['workflow_tokens']} tokens · {result['workflow_latency_s']:.2f}s · no loop, no decision-making")

        st.divider()
        speedup = (result["agent_latency_s"] / result["workflow_latency_s"]) if result["workflow_latency_s"] else 0
        token_ratio = (result["agent_tokens"] / result["workflow_tokens"]) if result["workflow_tokens"] else 0
        st.info(f"For this question: workflow was **{speedup:.1f}x faster** and used **{token_ratio:.1f}x fewer tokens** than the agent.")


def render_dispatcher_panel(client: BackendClient) -> None:
    """Week 7 extra: instead of running both systems to compare them, this
    runs ONLY whichever one the hybrid dispatcher (agents/dispatcher.py)
    decides fits the question -- the workflow for a simple single-employee
    lookup, the agent for anything whose path could vary (multiple
    employees, a comparison, a conditional). Shows the routing decision and
    why, not just the answer.
    """
    with st.expander("Week 7 - Hybrid Dispatcher", expanded=False):
        st.caption(
            "One question in, one system out -- the dispatcher picks the cheap fixed workflow "
            "or the adaptive agent loop per question, instead of always running both."
        )
        question = st.selectbox("Question", _SAMPLE_QUESTIONS, index=0, key="dispatch_question_select")
        custom = st.text_input("...or type your own (must name an employee, e.g. 'E3')", value="", key="dispatch_question_custom")
        final_question = custom.strip() or question

        if st.button("Ask via dispatcher", key="ask_dispatch_btn"):
            with st.spinner("Routing and running..."):
                try:
                    result = client.ask_dispatch(final_question)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    return
            st.session_state["dispatch_result"] = result

        result = st.session_state.get("dispatch_result")
        if not result:
            return

        st.success(f"Routed to: **{result['system_used']}** -- {result['reason']}")
        if result["fallback_used"]:
            st.warning("Fallback triggered: the workflow's fixed path couldn't handle this input, so the agent ran instead.")
        st.markdown(f"**Answer:** {result['answer'] or '_(none)_'}")
        st.caption(f"{result['total_tokens']} tokens · {result['wall_clock_s']:.2f}s · terminated: {result['terminated_reason'] or 'normally'}")
        if result["agent_steps"]:
            for i, step in enumerate(result["agent_steps"], 1):
                with st.container(border=True):
                    st.markdown(f"**Step {i}: `{step['action']}`**")
                    st.caption(f"Thought: {step['thought']}")
                    st.code(json.dumps(step["action_input"]), language="json")
                    st.caption(f"Observation: {step['observation']}")


def render_conversation_panel(client: BackendClient) -> None:
    """Week 7 bonus: a multi-turn chat with the agent, kept alive across
    turns by a session_id the backend uses to hold a sliding-window
    conversation (agents/conversation.py). The expander surfaces what's
    happening under the hood -- window size, running summary, whether
    jurisdiction is known -- since the whole point of this panel is to make
    the memory mechanism visible, not just the chat.
    """
    with st.expander("Week 7 - Agent Conversation Memory (bonus)", expanded=False):
        st.caption(
            "Multi-turn chat with the agent: it keeps a sliding window of recent turns plus a "
            "running summary of older ones, and persists only your jurisdiction across a backend restart."
        )
        if "conversation_session_id" not in st.session_state:
            st.session_state["conversation_session_id"] = str(uuid.uuid4())
        if "conversation_transcript" not in st.session_state:
            st.session_state["conversation_transcript"] = []

        col_id, col_reset = st.columns([4, 1])
        col_id.text_input("Session ID", value=st.session_state["conversation_session_id"], disabled=True, key="conversation_session_id_display")
        if col_reset.button("New session", key="conversation_new_session_btn"):
            st.session_state["conversation_session_id"] = str(uuid.uuid4())
            st.session_state["conversation_transcript"] = []
            st.rerun()

        for turn in st.session_state["conversation_transcript"]:
            st.chat_message("user").markdown(turn["question"])
            st.chat_message("assistant").markdown(turn["answer"] or "_(no answer)_")

        if question := st.chat_input("Ask the agent (multi-turn)...", key="conversation_chat_input"):
            st.chat_message("user").markdown(question)
            with st.spinner("Thinking..."):
                try:
                    result = client.ask_conversation(st.session_state["conversation_session_id"], question)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    return
            st.session_state["conversation_transcript"].append({"question": question, "answer": result["answer"]})
            st.session_state["conversation_last_result"] = result
            st.rerun()

        last = st.session_state.get("conversation_last_result")
        if last:
            st.divider()
            st.caption(
                f"Turn {last['turn_number']} · window={last['window_size']} turns · "
                f"jurisdiction known: {last['jurisdiction_known'] or 'not yet'} · "
                f"{'summarized this turn' if last['summarized_this_turn'] else 'no fold this turn'}"
            )
            if last["summary"]:
                st.caption(f"Running summary: {last['summary']}")


def render_results_showcase_panel(client: BackendClient) -> None:
    """Week 7 showcase: the real saved deliverables (race numbers, verdict,
    budget-termination proof, the live branching-case transcript, and
    persisted-jurisdiction proof), read straight from disk by the backend.
    Nothing here is re-run or computed on the fly -- nothing to click,
    just what actually happened, kept expanded by default since it's meant
    to be shown, not dug for.
    """
    with st.expander("Week 7 - Results Showcase", expanded=True):
        try:
            data = client.get_agent_results()
        except Exception as exc:
            st.error(f"Could not load results: {exc}")
            return

        st.subheader("Agent vs. Fixed Workflow (10 questions)")
        if data["race_rows"]:
            st.dataframe(data["race_rows"], use_container_width=True, hide_index=True)
            st.caption("Same pass rate both ways -- the workflow wins on speed and cost for this question set.")
        else:
            st.info("race.csv not found -- run scripts/race_agent_vs_workflow.py first.")

        st.subheader("Hybrid Dispatcher (10 official + 1 comparison question)")
        if data["dispatch_race_rows"]:
            st.dataframe(data["dispatch_race_rows"], use_container_width=True, hide_index=True)
            st.caption("All 10 single-employee questions routed to the workflow; the comparison question routed to the agent.")
        else:
            st.info("dispatch_race.csv not found -- run scripts/race_dispatcher.py first.")

        st.subheader("Verdict")
        if data["verdict"]:
            st.markdown(data["verdict"].replace("\n\n", "\n\n&nbsp;\n\n"))
        else:
            st.info("verdict.txt not found.")

        col_budget, col_branch = st.columns(2)
        with col_budget:
            st.subheader("Budget Termination Proof")
            if data["budget_termination_log"]:
                st.code(data["budget_termination_log"], language=None)
            else:
                st.info("budget_termination_log.txt not found.")
        with col_branch:
            st.subheader("Live Branching-Case Transcript")
            if data["branching_case_demo"]:
                st.code(data["branching_case_demo"], language=None)
            else:
                st.info("branching_case_demo.txt not found.")

        if data["third_tool_diff"]:
            st.subheader("Third Tool Diff")
            st.markdown(data["third_tool_diff"])

        st.subheader("Persisted Jurisdiction (survives a backend restart)")
        if data["persisted_sessions"]:
            rows = [{"session_id": sid, **record} for sid, record in data["persisted_sessions"].items()]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("Only jurisdiction survives a restart, by design -- window/summary do not.")
        else:
            st.info("No sessions persisted yet -- use the conversation panel above at least once.")
