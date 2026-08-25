from __future__ import annotations

import httpx

_DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class BackendClient:
    """Thin httpx wrapper around the FastAPI backend — pooled connections,
    explicit timeouts (embedding + Gemini calls are slower than a typical
    JSON API round trip).
    """

    def __init__(self, base_url: str) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=_DEFAULT_TIMEOUT)

    def upload_document(self, filename: str, content: bytes) -> dict:
        files = {"file": (filename, content, "application/pdf")}
        response = self._client.post("/documents/upload", files=files)
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> list[dict]:
        response = self._client.get("/documents")
        response.raise_for_status()
        return response.json()["documents"]

    def delete_document(self, doc_id: str) -> None:
        response = self._client.delete(f"/documents/{doc_id}")
        response.raise_for_status()

    def chat(
        self,
        query: str,
        use_keyword_search: bool,
        use_query_expansion: bool,
        generator_provider: str,
        judge_provider: str,
        generator_temperature: float,
        judge_temperature: float,
        top_k: int,
        doc_ids: list[str] | None,
    ) -> dict:
        payload = {
            "query": query,
            "use_keyword_search": use_keyword_search,
            "use_query_expansion": use_query_expansion,
            "generator_provider": generator_provider,
            "judge_provider": judge_provider,
            "generator_temperature": generator_temperature,
            "judge_temperature": judge_temperature,
            "top_k": top_k,
            "doc_ids": doc_ids,
        }
        response = self._client.post("/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
