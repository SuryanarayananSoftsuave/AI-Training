from __future__ import annotations

import json
import mimetypes
from typing import Iterator

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
        # The backend validates by file extension, not this content-type --
        # guessed here only so the multipart request isn't misleadingly
        # labeled "application/pdf" for a .docx/.xlsx/etc. upload.
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files = {"file": (filename, content, content_type)}
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

    def stream_chat(
        self,
        query: str,
        use_keyword_search: bool,
        use_query_expansion: bool,
        use_mmr: bool,
        generator_provider: str,
        judge_provider: str,
        generator_temperature: float,
        judge_temperature: float,
        top_k: int,
        doc_ids: list[str] | None,
    ) -> Iterator[tuple[str, str | dict]]:
        """Yields ("delta", text) for each streamed answer-text chunk, then
        exactly one ("final", dict) with the parsed final payload (citations,
        judgment, retrieval_debug). Hand-parses the backend's SSE stream --
        `httpx.Client.stream` supports this without needing an async client
        or a dedicated SSE library, since Streamlit's execution model is
        synchronous top-to-bottom per script run anyway.
        """
        payload = {
            "query": query,
            "use_keyword_search": use_keyword_search,
            "use_query_expansion": use_query_expansion,
            "use_mmr": use_mmr,
            "generator_provider": generator_provider,
            "judge_provider": judge_provider,
            "generator_temperature": generator_temperature,
            "judge_temperature": judge_temperature,
            "top_k": top_k,
            "doc_ids": doc_ids,
        }
        with self._client.stream("POST", "/chat", json=payload) as response:
            response.raise_for_status()
            event_type: str | None = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:") and event_type is not None:
                    data = json.loads(line[len("data:"):].strip())
                    yield (event_type, data["text"] if event_type == "delta" else data)
                    event_type = None

    def health(self) -> dict:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
