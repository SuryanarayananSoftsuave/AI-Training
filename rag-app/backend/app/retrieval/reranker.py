from __future__ import annotations

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(model_name: str, query: str, candidates: list[str]) -> list[float]:
    """Cross-encoder relevance scores, sigmoid-normalized to [0, 1].

    These scores are only meaningful as a *within-query* ranking signal —
    never compare them across different queries or treat one as a fixed
    absolute relevance threshold.
    """
    if not candidates:
        return []
    model = get_reranker(model_name)
    pairs = [[query, text] for text in candidates]
    raw_scores = model.predict(pairs)
    return [_sigmoid(float(s)) for s in raw_scores]
