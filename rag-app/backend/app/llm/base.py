from __future__ import annotations

from typing import AsyncIterator, Protocol

from app.models.schemas import Judgment


class LLMClient(Protocol):
    """Every LLM provider client (Gemini, Groq, ...) implements this same
    shape, so `chat_service.py` can pick a provider per-request without
    caring which one it actually is.

    `stream_answer` yields raw answer-text chunks only -- no structured
    schema, since JSON-schema-constrained output and token streaming are
    mutually exclusive on both providers. Which numbered sources the
    generator actually cited is recovered by `chat_service.py` via a regex
    over the fully-accumulated text once the stream ends, relying on the
    prompt's existing inline-citation instruction (`[1]`, `[2]`, ...).

    `prompt_version` defaults to `None` (the current live template) for
    normal chat traffic; trace replay (`scripts/replay_trace.py`) passes the
    trace's recorded version explicitly, so a historical trace always
    replays against the exact prompt it was generated with, even after a
    newer version has since gone live.
    """

    async def expand_query(self, question: str, variant_count: int, temperature: float) -> list[str]: ...

    def stream_answer(
        self, question: str, numbered_context: str, temperature: float, prompt_version: str | None = None
    ) -> AsyncIterator[str]: ...

    async def judge_answer(self, question: str, answer: str, numbered_context: str, temperature: float) -> Judgment: ...

    @property
    def generator_model_name(self) -> str: ...

    @property
    def judge_model_name(self) -> str: ...
