"""Week 7 bonus (W7-Task-Set-C.md Sec.5): sliding-window conversation memory
for the ReAct agent, plus lossy summarization of what falls out of the
window.

Deliberately kept separate from react_agent.py's own state: run_agent stays
a pure per-question function that only *reads* a ConversationState (to
build a richer prompt) and, via its caller, *writes* the one fact
(jurisdiction) that must survive a restart. Window/summary mutation is the
CALLER's job (append_turn, below) -- this keeps every existing single-shot
caller of run_agent (race_agent_vs_workflow.py, run_agent_demo.py, the
existing /agents/ask endpoint) completely unaffected when no conversation is
passed.

Only question+answer are kept per turn, not the internal thought/action/
observation steps -- the window is meant to let the agent recall what was
already discussed, not to replay its own scratch reasoning back to itself.

Folding happens in batches (FOLD_BATCH_SIZE turns at a time) rather than one
at a time: fewer summarization calls, and a bigger fold is a more honest
test of what a real lossy summarizer actually discards (which is exactly
what the bonus asks us to go find and name).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from groq import AsyncGroq


@dataclass
class Turn:
    question: str
    answer: str | None


@dataclass
class ConversationState:
    session_id: str
    window: list[Turn] = field(default_factory=list)
    summary: str = ""
    jurisdiction: str | None = None
    employee_id: str | None = None
    turn_count: int = 0


_SUMMARY_PROMPT_TEMPLATE = """You are maintaining a running summary of an ongoing HR conversation so \
older turns can be dropped from the active context without losing the important facts.

Existing summary (may be empty if this is the first fold):
{prior_summary}

Turns to fold into the summary:
{turns_text}

Write an updated summary (2-4 sentences) that preserves the important facts from both the \
existing summary and the new turns above -- employee identities, numbers, and anything the \
employee explicitly asked to be remembered. Be concise; it is fine to drop incidental detail \
that isn't load-bearing for future questions."""


def _render_turns(turns: list[Turn]) -> str:
    return "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in turns)


async def summarize_dropped_turns(client: AsyncGroq, model: str, prior_summary: str, dropped: list[Turn]) -> str:
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(prior_summary=prior_summary or "(none yet)", turns_text=_render_turns(dropped))
    response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return response.choices[0].message.content or prior_summary


async def append_turn(
    state: ConversationState, turn: Turn, client: AsyncGroq, model: str, window_turns: int, fold_batch_size: int,
) -> bool:
    """Adds `turn` to the window; if the window now exceeds `window_turns`,
    folds the oldest `fold_batch_size` turns into `state.summary` via one
    real summarization call. Returns True if a fold happened this call.
    """
    state.window.append(turn)
    state.turn_count += 1
    if len(state.window) <= window_turns:
        return False
    dropped, state.window[:] = state.window[:fold_batch_size], state.window[fold_batch_size:]
    state.summary = await summarize_dropped_turns(client, model, state.summary, dropped)
    return True


def render_conversation_block(state: ConversationState | None) -> str:
    if state is None:
        return ""
    parts = []
    if state.jurisdiction:
        parts.append(f"Known fact carried over from this employee's session (possibly from before a restart): jurisdiction={state.jurisdiction!r}.")
    if state.summary:
        parts.append(f"Summary of earlier turns in this conversation: {state.summary}")
    if state.window:
        parts.append("Recent turns in this conversation:\n" + _render_turns(state.window))
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"
