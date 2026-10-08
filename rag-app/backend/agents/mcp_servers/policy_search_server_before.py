"""The deliberate "before" version of policy_search_server.py's search_handbook
tool, for requirement #5's required before/after transcript only -- never
registered in mcp_config_one_server.json / mcp_config_two_servers.json, so it
never participates in the real discovery/tool-count story.

Same one-line docstring and same swallowed-error style this tool actually had
before this week's rewrite: `as_of` is accepted (so the same question can be
asked) but any value at all just raises a bare "not found", indistinguishable
from "the server is down" or "you misspelled the tool name" -- exactly the
"Swallowing the effective-date miss into 'Error: not found'" common mistake
W9-Task-Set-C.md calls out. The real search itself still calls back into the
main FastAPI process's /mcp/internal/search-handbook (see
policy_search_server.py's module docstring for why), so the "before" and
"after" retrieval behavior is identical -- only the docstring and the as_of
error path differ, which is the actual point being contrasted.

Run standalone: `python -m agents.mcp_servers.policy_search_server_before`
"""
from __future__ import annotations

import httpx
from fastmcp import FastMCP

from app.core.config import Settings

_settings = Settings()

mcp = FastMCP("policy-search")


@mcp.tool()
async def search_handbook(query: str, as_of: str | None = None) -> dict:
    """Search the HR policy handbook for a topic and return a passage."""
    if as_of is not None:
        raise ValueError("not found")

    async with httpx.AsyncClient(base_url=_settings.backend_internal_url, timeout=60.0) as http:
        response = await http.post("/mcp/internal/search-handbook", json={"query": query})
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
