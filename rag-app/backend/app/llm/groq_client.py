from __future__ import annotations

import json
from typing import AsyncIterator

from groq import AsyncGroq

from app.llm.prompts import render_generation, render_judge, render_query_expansion
from app.llm.verification import verify_and_score_claims
from app.models.schemas import JudgeVerdict, Judgment

# Groq's structured-output mode requires a fully inlined JSON schema (no
# $ref/$defs) with every property marked required and additionalProperties
# false at every level -- hand-written here rather than derived from a
# Pydantic model's auto-generated (ref-based) schema, since strict-mode
# support for nested $refs varies by provider and isn't worth risking.
_JUDGE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["grounded", "partially_grounded", "hallucinated"]},
                "confidence": {"type": "integer"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "supported": {"type": "boolean"},
                            "evidence_quote": {"type": "string"},
                        },
                        "required": ["claim", "supported", "evidence_quote"],
                        "additionalProperties": False,
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["verdict", "confidence", "claims", "notes"],
            "additionalProperties": False,
        },
    },
}

_QUERY_VARIANTS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_variants",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
            "required": ["variants"],
            "additionalProperties": False,
        },
    },
}


class GroqClient:
    """Same interface as GeminiClient (app/llm/base.py's LLMClient Protocol)
    -- a second, independent LLM provider, selectable per-request from the
    UI, so generation/judging can be compared across providers rather than
    only across models within one provider.
    """

    def __init__(self, api_key: str, generator_model: str, judge_model: str) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._generator_model = generator_model
        self._judge_model = judge_model

    @property
    def generator_model_name(self) -> str:
        return self._generator_model

    @property
    def judge_model_name(self) -> str:
        return self._judge_model

    async def _call(self, model: str, prompt: str, schema: dict, temperature: float) -> dict:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content or "{}")

    async def expand_query(self, question: str, variant_count: int, temperature: float) -> list[str]:
        data = await self._call(self._generator_model, render_query_expansion(question, variant_count), _QUERY_VARIANTS_SCHEMA, temperature)
        return data.get("variants", [])

    async def stream_answer(
        self, question: str, numbered_context: str, temperature: float, prompt_version: str | None = None
    ) -> AsyncIterator[str]:
        """No `response_format` here -- structured JSON output and token
        streaming don't combine on this API either. Which sources got cited
        is recovered by `chat_service.py` via regex over the accumulated
        text, relying on the prompt's inline-citation rule.
        """
        stream = await self._client.chat.completions.create(
            model=self._generator_model,
            messages=[{"role": "user", "content": render_generation(question, numbered_context, version=prompt_version)}],
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def judge_answer(self, question: str, answer: str, numbered_context: str, temperature: float) -> Judgment:
        data = await self._call(self._judge_model, render_judge(question, answer, numbered_context), _JUDGE_OUTPUT_SCHEMA, temperature)

        raw_claims = [(c["claim"], c["supported"], c["evidence_quote"]) for c in data.get("claims", [])]
        verified_claims, confidence = verify_and_score_claims(raw_claims, numbered_context, fallback_confidence=data.get("confidence", 0))

        try:
            verdict = JudgeVerdict(data["verdict"])
        except (ValueError, KeyError):
            verdict = JudgeVerdict.PARTIALLY_GROUNDED

        return Judgment(verdict=verdict, confidence=confidence, claims=verified_claims, notes=data.get("notes", ""))
