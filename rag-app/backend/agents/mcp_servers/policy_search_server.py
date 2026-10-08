"""Week 9 (W9-Task-Set-C.md): our own MCP server, exposing one real capability
of this app -- policy search -- over the standard protocol so any MCP client
can discover and call it, not just our own hand-rolled agent.

The actual retrieval (retrieve_candidates/rerank_candidates) runs inside the
main FastAPI process, via a loopback call to its own
/mcp/internal/search-handbook endpoint (app/api/mcp.py) -- NOT re-loaded
in this subprocess. Loading a second copy of the embedder + cross-encoder
reranker here was tried first and hit a real `OSError: The paging file is
too small for this operation to complete` on a dev machine already running
the full stack (backend + frontend + the Docker Compose services) -- a real
resource ceiling, not a hypothetical one. Routing through the already-loaded
models in the main process is also, arguably, the more honest MCP design:
this server is a thin protocol adapter in front of a capability the app
already serves, not a second copy of the app's ML pipeline.

Requirement #5 (20 pts) lives in THIS file, on OUR OWN server, as the rubric
requires: `as_of` is an optional policy-version date checked locally (cheap
JSON read, no models) against the real `upload_timestamp` of already-indexed
documents (DocumentRegistry -- genuine data, not invented). Asking for a
version older than anything ever indexed raises a plain, informative
ValueError; FastMCP converts that into a recoverable
`CallToolResult(is_error=True, ...)` automatically, with zero extra
error-handling code needed (verified in WEEK9_MCP_RESEARCH.md).
`policy_search_server_before.py` is the deliberately-worse "before" version
of this same tool, kept only for the required before/after transcript.

Run standalone: `python -m agents.mcp_servers.policy_search_server`
(stdio transport -- spawned as a subprocess by whatever MCP client connects;
requires the main FastAPI backend to already be running).
"""
from __future__ import annotations

from datetime import date

import httpx
from fastmcp import FastMCP

from app.core.config import Settings
from app.models.schemas import DocumentStatus
from app.registry.json_store import DocumentRegistry

_settings = Settings()
_registry = DocumentRegistry(_settings.registry_path)

mcp = FastMCP("policy-search")


def _earliest_indexed_date() -> date | None:
    indexed = [r for r in _registry.list_all() if r.status == DocumentStatus.INDEXED]
    if not indexed:
        return None
    return min(r.upload_timestamp for r in indexed).date()


@mcp.tool()
async def search_handbook(query: str, as_of: str | None = None) -> dict:
    """Search the HR policy handbook for a topic and return the single most
    relevant passage, its source document, and section heading.

    Call this whenever you need real policy text to answer a question --
    never answer from general knowledge about what an HR policy "usually"
    says. Pass `as_of` (an ISO date, e.g. "2024-06-01") only when the
    question is specifically about a historical policy version; omit it to
    search whatever is currently indexed. If `as_of` predates every version
    that has ever been indexed, this tool raises an error naming the
    earliest version that actually exists -- that error IS the answer
    ("no such version was ever indexed"), not a failure to retry around.
    """
    if as_of is not None:
        try:
            requested = date.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError(f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}") from exc
        earliest = _earliest_indexed_date()
        if earliest is not None and requested < earliest:
            raise ValueError(f"no policy version effective {as_of}: earliest indexed is {earliest.isoformat()}")

    async with httpx.AsyncClient(base_url=_settings.backend_internal_url, timeout=60.0) as http:
        response = await http.post("/mcp/internal/search-handbook", json={"query": query})
        response.raise_for_status()
        data = response.json()

    data["as_of"] = as_of
    return data


if __name__ == "__main__":
    mcp.run(transport="stdio")
