from __future__ import annotations

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.llm.prompts import build_generation_prompt, build_judge_prompt, build_query_expansion_prompt
from app.llm.verification import verify_and_score_claims
from app.models.schemas import ClaimVerdict, JudgeVerdict, Judgment


class _QueryVariants(BaseModel):
    variants: list[str]


class _GeneratedAnswer(BaseModel):
    answer: str
    used_source_indices: list[int]


class _JudgeOutput(BaseModel):
    verdict: str
    confidence: int
    claims: list[ClaimVerdict]
    notes: str


class GeminiClient:
    """Two independent Gemini calls: `generate_answer` (the RAG answer,
    grounded strictly in the numbered context) and `judge_answer` (a
    separate, differently-tiered model that fact-checks the answer against
    that same context). Running the judge on a different model tier than
    the generator mitigates the documented self-preference bias of
    same-family LLM judges.
    """

    def __init__(self, api_key: str, generator_model: str, judge_model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._generator_model = generator_model
        self._judge_model = judge_model

    def expand_query(self, question: str, variant_count: int, temperature: float) -> list[str]:
        """Alternate phrasings of the same question, for multi-query retrieval --
        run on the (cheaper, faster) generator model since this is a simple
        rewrite task, not a task needing the judge's separate tier.
        """
        response = self._client.models.generate_content(
            model=self._generator_model,
            contents=build_query_expansion_prompt(question, variant_count),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_QueryVariants,
                temperature=temperature,
            ),
        )
        parsed: _QueryVariants = response.parsed
        return parsed.variants

    def generate_answer(self, question: str, numbered_context: str, temperature: float) -> tuple[str, list[int]]:
        response = self._client.models.generate_content(
            model=self._generator_model,
            contents=build_generation_prompt(question, numbered_context),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GeneratedAnswer,
                temperature=temperature,
            ),
        )
        parsed: _GeneratedAnswer = response.parsed
        return parsed.answer, parsed.used_source_indices

    def judge_answer(self, question: str, answer: str, numbered_context: str, temperature: float) -> Judgment:
        response = self._client.models.generate_content(
            model=self._judge_model,
            contents=build_judge_prompt(question, answer, numbered_context),
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
