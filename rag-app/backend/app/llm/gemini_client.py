from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.llm.prompts import render_generation, render_judge, render_query_expansion
from app.llm.verification import verify_and_score_claims
from app.models.schemas import ClaimVerdict, JudgeVerdict, Judgment


class _QueryVariants(BaseModel):
    variants: list[str]


class _JudgeOutput(BaseModel):
    verdict: str
    confidence: int
    claims: list[ClaimVerdict]
    notes: str


class GeminiClient:
    """Two independent Gemini calls: `stream_answer` (the RAG answer,
    grounded strictly in the numbered context, streamed token-by-token) and
    `judge_answer` (a separate, differently-tiered model that fact-checks
    the answer against that same context). Running the judge on a different
    model tier than the generator mitigates the documented self-preference
    bias of same-family LLM judges.
    """

    def __init__(self, api_key: str, generator_model: str, judge_model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._generator_model = generator_model
        self._judge_model = judge_model

    @property
    def generator_model_name(self) -> str:
        return self._generator_model

    @property
    def judge_model_name(self) -> str:
        return self._judge_model

    async def expand_query(self, question: str, variant_count: int, temperature: float) -> list[str]:
        """Alternate phrasings of the same question, for multi-query retrieval --
        run on the (cheaper, faster) generator model since this is a simple
        rewrite task, not a task needing the judge's separate tier.
        """
        response = await self._client.aio.models.generate_content(
            model=self._generator_model,
            contents=render_query_expansion(question, variant_count),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_QueryVariants,
                temperature=temperature,
            ),
        )
        parsed: _QueryVariants = response.parsed
        return parsed.variants

    async def stream_answer(
        self, question: str, numbered_context: str, temperature: float, prompt_version: str | None = None
    ) -> AsyncIterator[str]:
        """No `response_schema` here -- structured JSON output and token
        streaming are mutually exclusive on this API. Which sources got
        cited is recovered by `chat_service.py` via regex over the
        accumulated text, relying on the prompt's inline-citation rule.
        """
        stream = await self._client.aio.models.generate_content_stream(
            model=self._generator_model,
            contents=render_generation(question, numbered_context, version=prompt_version),
            config=types.GenerateContentConfig(temperature=temperature),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def judge_answer(self, question: str, answer: str, numbered_context: str, temperature: float) -> Judgment:
        response = await self._client.aio.models.generate_content(
            model=self._judge_model,
            contents=render_judge(question, answer, numbered_context),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_JudgeOutput,
                temperature=temperature,
            ),
        )
        parsed: _JudgeOutput = response.parsed

        raw_claims = [(c.claim, c.supported, c.evidence_quote) for c in parsed.claims]
        verified_claims, confidence = verify_and_score_claims(raw_claims, numbered_context, fallback_confidence=parsed.confidence)

        try:
            verdict = JudgeVerdict(parsed.verdict)
        except ValueError:
            verdict = JudgeVerdict.PARTIALLY_GROUNDED

        return Judgment(verdict=verdict, confidence=confidence, claims=verified_claims, notes=parsed.notes)
