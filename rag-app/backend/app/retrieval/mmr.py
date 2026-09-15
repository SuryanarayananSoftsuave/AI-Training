from __future__ import annotations


def mmr_select(relevance_scores: list[float], vectors: list[list[float]], k: int, lambda_mult: float = 0.5) -> list[int]:
    """Maximal Marginal Relevance: greedily selects `k` indices from the pool
    that balance relevance to the query (`relevance_scores`) against
    diversity from what's already been selected (cosine similarity between
    `vectors`, which must be pre-normalized -- dot product == cosine, same
    convention already used in `chat_service._filter_variants_by_similarity`).

    At each step, picks argmax(lambda_mult * relevance[i] - (1 - lambda_mult)
    * max_similarity_to_already_selected). The first pick is always the
    single highest-relevance candidate (no prior selection to be similar
    to), so callers relying on "the best score" for a threshold check can
    read it from the pre-MMR sorted pool without needing to special-case
    this function's first output.

    lambda_mult close to 1.0 behaves like a plain top-k relevance cut;
    close to 0.0 prioritizes diversity almost regardless of relevance.
    """
    n = len(relevance_scores)
    k = min(k, n)
    if k <= 0:
        return []

    selected: list[int] = []
    remaining = set(range(n))

    while len(selected) < k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            max_similarity_to_selected = (
                max(sum(a * b for a, b in zip(vectors[idx], vectors[j])) for j in selected) if selected else 0.0
            )
            score = lambda_mult * relevance_scores[idx] - (1 - lambda_mult) * max_similarity_to_selected
            if score > best_score:
                best_idx, best_score = idx, score
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected
