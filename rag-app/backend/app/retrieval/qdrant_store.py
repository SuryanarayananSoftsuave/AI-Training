from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models


class QdrantStore:
    """Thin wrapper around a single Qdrant collection holding two named
    vectors per point: `dense` (semantic embedding) and `sparse` (BM25-style
    keyword vector). Hybrid search fuses both branches server-side via RRF;
    semantic-only search queries the dense vector alone.
    """

    _PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
        ("filename", models.PayloadSchemaType.KEYWORD),
        ("file_hash", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("page_number", models.PayloadSchemaType.INTEGER),
        ("upload_date", models.PayloadSchemaType.DATETIME),
    )

    def __init__(self, url: str, collection: str, dense_dim: int) -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection
        self._dense_dim = dense_dim

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"dense": models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field_name, schema in self._PAYLOAD_INDEXES:
            # wait=False: this only ever runs once, against a brand-new EMPTY
            # collection, before any upload/search can possibly happen -- so
            # blocking for each index to fully build serves no purpose here,
            # and doing so is actively harmful on a slow filesystem (e.g. a
            # Docker Desktop bind-mount on Windows): Qdrant's own REST layer
            # enforces a fixed ~5s client_request_timeout server-side, and on
            # slow disk I/O a `wait=True` payload-index build can exceed that
            # and hard-fail the whole app startup (observed in practice: 5
            # sequential index creates on an empty collection took 2.1s,
            # 3.4s, 3.9s, 3.9s, then the 5th was killed mid-request).
            await self._client.create_payload_index(
                self._collection, field_name=field_name, field_schema=schema, wait=False
            )

    async def is_connected(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def collection_exists(self) -> bool:
        return await self._client.collection_exists(self._collection)

    async def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def get_by_ids(self, ids: list[str]) -> list[models.Record]:
        """Fetches specific points by ID with payload (chunk text included) --
        used by trace replay to reconstruct a historical `numbered_context`
        from the lean `chunk_id`s a trace actually stores. A point missing
        from the result (deleted or re-ingested since the trace was
        recorded) is simply absent, not an error -- the caller decides how
        to report that as "could not be reconstructed."
        """
        return await self._client.retrieve(collection_name=self._collection, ids=ids, with_payload=True)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
        )

    async def search(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector | None,
        limit: int,
        doc_ids: list[str] | None,
        with_vectors: bool = False,
    ) -> list[models.ScoredPoint]:
        """`sparse_vector is None` -> semantic-only dense search.
        `sparse_vector` present -> hybrid search, both branches prefetched
        and fused server-side with Reciprocal Rank Fusion.

        `with_vectors` is only needed by callers doing their own vector math
        downstream (e.g. MMR re-selection) -- left `False` by default so the
        normal chat path doesn't pay to ship vectors back over the wire.
        """
        doc_filter = (
            models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
            if doc_ids
            else None
        )

        if sparse_vector is None:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using="dense",
                query_filter=doc_filter,
                limit=limit,
                with_vectors=with_vectors,
            )
            return response.points

        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=limit, filter=doc_filter),
            models.Prefetch(query=sparse_vector, using="sparse", limit=limit, filter=doc_filter),
        ]
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_vectors=with_vectors,
        )
        return response.points
