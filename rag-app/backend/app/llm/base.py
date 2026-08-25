from __future__ import annotations

from typing import Protocol

from app.models.schemas import Judgment


class LLMClient(Protocol):
    """Every LLM provider client (Gemini, Groq, ...) implements this same
    shape, so `chat_service.py` can pick a provider per-request without
    caring which one it actually is.
    """

    def expand_query(self, question: str, variant_count: int, temperature: float) -> list[str]: ...

    def generate_answer(self, question: str, numbered_context: str, temperature: float) -> tuple[str, list[int]]: ...

    def judge_answer(self, question: str, answer: str, numbered_context: str, temperature: float) -> Judgment: ...
