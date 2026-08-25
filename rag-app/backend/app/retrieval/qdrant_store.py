from __future__ import annotations

from qdrant_client import QdrantClient, models


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
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dense_dim = dense_dim

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"dense": models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field_name, schema in self._PAYLOAD_INDEXES:
            self._client.create_payload_index(self._collection, field_name=field_name, field_schema=schema)

    def is_connected(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
        )

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector | None,
        limit: int,
        doc_ids: list[str] | None,
    ) -> list[models.ScoredPoint]:
        """`sparse_vector is None` -> semantic-only dense search.
        `sparse_vector` present -> hybrid search, both branches prefetched
        and fused server-side with Reciprocal Rank Fusion.
        """
        doc_filter = (
            models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
            if doc_ids
            else None
        )

        if sparse_vector is None:
            response = self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using="dense",
                query_filter=doc_filter,
                limit=limit,
            )
            return response.points

        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=limit, filter=doc_filter),
            models.Prefetch(query=sparse_vector, using="sparse", limit=limit, filter=doc_filter),
        ]
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return response.points
