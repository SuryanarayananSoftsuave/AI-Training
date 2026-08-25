from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformer:
    """Loaded once per process and cached — embedding models are too heavy
    to construct on every request or ingestion call.
    """
    return SentenceTransformer(model_name)


def embed_passages(model_name: str, texts: list[str]) -> list[list[float]]:
    """Passage-side embeddings: Qwen3-Embedding needs no prefix on this side."""
    model = get_embedder(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(model_name: str, text: str) -> list[float]:
    """Query-side embedding: `prompt_name="query"` applies the model's own
    shipped instruction template rather than a hand-built prefix string,
    which is the documented failure-prone part of this step.
    """
    model = get_embedder(model_name)
    vector = model.encode(text, prompt_name="query", normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
