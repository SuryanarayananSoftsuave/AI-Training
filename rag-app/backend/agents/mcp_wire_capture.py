"""Week 9 (W9-Task-Set-C.md) requirement #4 (25 pts): captures the raw
JSON-RPC initialize -> tools/list -> tools/call exchange against ONE MCP
server, by wrapping the anyio streams the low-level `mcp` SDK's stdio_client
hands back BEFORE constructing ClientSession. Proven in this project's
scratch-venv research (WEEK9_MCP_RESEARCH.md) to capture every real frame in
order -- 7 frames for one tool call: initialize request/response,
notifications/initialized, tools/list request/response, tools/call
request/response.

Deliberately uses the low-level `mcp` SDK here, not `fastmcp.Client` -- the
raw anyio streams are only exposed at that layer. `fastmcp.Client`'s
multi-server aggregation (used by agents/mcp_agent.py) abstracts this
transport detail away entirely, which is exactly why it's the wrong tool for
this specific requirement.
"""
from __future__ import annotations

from typing import Literal

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.mcp_config import load_config

ServerName = Literal["policy-search", "hris"]


def _server_params(server: ServerName) -> StdioServerParameters:
    # Both servers are always defined in the two-server config, regardless
    # of which one the live agent is currently connected with -- this
    # capture utility is a standalone diagnostic, not routed through the
    # agent's own connection state.
    config = load_config("two")
    spec = config["mcpServers"][server]
    return StdioServerParameters(command=spec["command"], args=spec["args"], cwd=spec["cwd"])


async def capture(server: ServerName, tool_name: str, tool_args: dict) -> list[dict]:
    """Runs a real initialize -> tools/list -> tools/call sequence against
    `server`, returning the ordered list of raw JSON-RPC frames as
    {"direction": "client->server" | "server->client", "frame": {...}}.
    """
    wire_log: list[dict] = []

    def _record(direction: str, session_message) -> None:
        msg = session_message.message
        wire_log.append({"direction": direction, "frame": msg.model_dump(mode="json", exclude_none=True)})

    async def _capturing_read_stream(raw_read, send_stream):
        async with raw_read:
            async for item in raw_read:
                if not isinstance(item, Exception):
                    _record("server->client", item)
                await send_stream.send(item)

    async def _capturing_write_stream(raw_write, receive_stream):
        async with raw_write:
            async for item in receive_stream:
                _record("client->server", item)
                await raw_write.send(item)

    params = _server_params(server)
    async with stdio_client(params) as (raw_read, raw_write):
        read_send, read_recv = anyio.create_memory_object_stream(100)
        write_send, write_recv = anyio.create_memory_object_stream(100)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_capturing_read_stream, raw_read, read_send)
            tg.start_soon(_capturing_write_stream, raw_write, write_recv)

            async with ClientSession(read_recv, write_send) as session:
                await session.initialize()
                await session.list_tools()
                await session.call_tool(tool_name, tool_args)

            tg.cancel_scope.cancel()

    return wire_log
