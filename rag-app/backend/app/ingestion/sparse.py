from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import models


@lru_cache(maxsize=1)
def _get_sparse_model(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


def embed_sparse(model_name: str, texts: list[str]) -> list[models.SparseVector]:
    """Local, network-free BM25-style sparse vectors — this is the 'keyword
    search' leg of hybrid retrieval, fused server-side in Qdrant via RRF.
    """
    if not texts:
        return []
    model = _get_sparse_model(model_name)
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in model.embed(texts)]


def embed_sparse_query(model_name: str, text: str) -> models.SparseVector:
    return embed_sparse(model_name, [text])[0]
