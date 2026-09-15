from __future__ import annotations

import asyncio

from qdrant_client import models

from app.core.config import Settings
from app.ingestion.embedder import embed_query
from app.ingestion.sparse import embed_sparse_query
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.reranker import rerank


async def _retrieve_for_query(
    query_text: str,
    use_keyword_search: bool,
    limit: int,
    doc_ids: list[str] | None,
    settings: Settings,
    store: QdrantStore,
    with_vectors: bool = False,
) -> list[models.ScoredPoint]:
    """Embed + search for a single query string -- dense-only, or dense+sparse
    fused with RRF when hybrid search is on. Runs once per query variant.

    Embedding is CPU-bound (sentence-transformers/fastembed have no native
    async surface), so it's offloaded to a worker thread; the Qdrant search
    itself is a real async network call via `AsyncQdrantClient`.

    `with_vectors` is only needed when MMR re-selection is enabled (it does
    its own vector-similarity math downstream) -- left `False` by default so
    the normal chat path doesn't pay to ship vectors back over the wire.
    """
    dense_vector = await asyncio.to_thread(embed_query, settings.embedding_model_name, query_text)
    sparse_vector = (
        await asyncio.to_thread(embed_sparse_query, settings.sparse_model_name, query_text)
        if use_keyword_search
        else None
    )
    return await store.search(
        dense_vector=dense_vector, sparse_vector=sparse_vector, limit=limit, doc_ids=doc_ids, with_vectors=with_vectors
    )


async def retrieve_candidates(
    queries: list[str],
    use_keyword_search: bool,
    limit: int,
    doc_ids: list[str] | None,
    settings: Settings,
    store: QdrantStore,
    with_vectors: bool = False,
) -> tuple[list[models.ScoredPoint], int]:
    """Retrieve for every query variant IN PARALLEL (`asyncio.gather` --
    each variant still does its own embed+search), then dedupe by Qdrant
    point ID. Returns (deduped candidates, total raw count before dedup) so
    the caller can still log/report both numbers.
    """
    per_query_results = await asyncio.gather(
        *[_retrieve_for_query(q, use_keyword_search, limit, doc_ids, settings, store, with_vectors) for q in queries]
    )

    seen_ids: set = set()
    candidates: list[models.ScoredPoint] = []
    total_raw = 0
    for result_list in per_query_results:
        total_raw += len(result_list)
        for point in result_list:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                candidates.append(point)

    return candidates, total_raw


async def rerank_candidates(
    reranker_model_name: str,
    query: str,
    candidates: list[models.ScoredPoint],
    top_k: int,
) -> list[tuple[models.ScoredPoint, float]]:
    """Cross-encoder rerank, always against the ORIGINAL question (never a
    query-expansion variant). CPU-bound, offloaded to a worker thread.
    """
    texts = [candidate.payload["text"] for candidate in candidates]
    scores = await asyncio.to_thread(rerank, reranker_model_name, query, texts)
    return sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
