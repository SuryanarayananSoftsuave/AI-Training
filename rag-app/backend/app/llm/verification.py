from __future__ import annotations

from app.models.schemas import ClaimVerdict


def verify_and_score_claims(
    raw_claims: list[tuple[str, bool, str]],
    numbered_context: str,
    fallback_confidence: int,
) -> tuple[list[ClaimVerdict], int]:
    """Shared judge post-processing, used by every LLM provider's client.

    Never trusts a judge's self-reported `supported` flag on its own -- a
    judge can hallucinate its own evidence. Downgrades any claim whose quote
    doesn't literally occur in the context, then recomputes confidence
    deterministically from the verified claims rather than trusting the
    model's self-reported number. `fallback_confidence` only applies when
    the judge returned zero claims at all.
    """
    verified = [
        ClaimVerdict(
            claim=claim,
            supported=supported and evidence_quote.strip() in numbered_context,
            evidence_quote=evidence_quote,
        )
        for claim, supported, evidence_quote in raw_claims
    ]
    if not verified:
        return verified, fallback_confidence
    confidence = round(100 * sum(c.supported for c in verified) / len(verified))
    return verified, confidence
