# Wire annotations — W9-Task-Set-C.md requirement #4 (25 pts)

Hand-annotated field-by-field walkthrough of the real 7-frame `initialize -> tools/list ->
tools/call` exchange captured live by `agents/mcp_wire_capture.py` against our own
`policy-search` server (`agents/mcp_servers/policy_search_server.py`, tool `search_handbook`).
A capture against `hris` produces the identical shape with different tool/content values — the
protocol-level fields below don't change per tool or per server.

## Where the model call happens, and where it doesn't (one line)

**The model runs on our side (the host/client), inside `agents/mcp_agent.py`'s loop, between the
`tools/list` response and the next `tools/call` request — never inside the MCP server process,
which only ever executes a plain Python function and returns its return value; it has no idea an
LLM exists, let alone which one is calling it.**

## Frame 1 — `client -> server`: `initialize` request

| Field | Meaning |
|---|---|
| `jsonrpc` | Always `"2.0"` — every MCP message is a JSON-RPC 2.0 envelope. |
| `id` | Correlates this request with its matching response (`id: 1` below). MCP is request/response over a single stdio pipe, so IDs are how concurrent in-flight calls stay distinguishable. |
| `method` | `"initialize"` — the first message on any new connection; nothing else may be sent before this handshake completes. |
| `params.protocolVersion` | The MCP wire-protocol version this client speaks — the server may downgrade/reject if it doesn't support it. |
| `params.capabilities` | What optional protocol features the *client* supports (sampling, roots, elicitation, etc.) — empty here since our client uses none of them. |
| `params.clientInfo` | Name/version of the connecting client library (`mcp`, `0.1.0`) — identifies the SDK, not the LLM or the person. |

## Frame 2 — `server -> client`: `initialize` response

| Field | Meaning |
|---|---|
| `result.protocolVersion` | The version the server agreed to use for this session. |
| `result.capabilities.tools.listChanged` | The server promises to notify us if its tool list ever changes at runtime (it can, in principle, add/remove tools live — we don't rely on that here). |
| `result.serverInfo` | Name/version of the server we connected to (`policy-search`, the FastMCP version) — this is how a client could tell which of several servers behind a gateway actually answered. |

## Frame 3 — `client -> server`: `notifications/initialized`

A one-way notification (no `id`, no response expected) confirming the handshake is complete and
the server may now accept `tools/list`/`tools/call`. This is the one frame that isn't a
request/response pair.

## Frame 4 — `client -> server`: `tools/list` request

Empty `params` (`_meta` only) — this call takes no arguments; it always means "tell me everything
you expose right now."

## Frame 5 — `server -> client`: `tools/list` response

| Field | Meaning |
|---|---|
| `result.tools[].name` | The exact string the model must use to invoke this tool — this is what `agents/mcp_agent.py` builds its Groq function-calling schema from at runtime; it is never a Python constant in that file. |
| `result.tools[].description` | **This is the docstring, verbatim** — requirement #5's "rewrite a docstring as a prompt" means editing exactly this string; it is the only thing telling the model when/why/how to call the tool. |
| `result.tools[].inputSchema` | A real JSON Schema auto-generated from the Python function's type hints (`query: str`, `as_of: str \| None = None`) — this is what constrains/validates the arguments the model is allowed to send; we never hand-wrote it. |
| `result.tools[].outputSchema` | What shape the return value takes — `additionalProperties: true` here because `search_handbook` returns a plain untyped `dict`, not a dataclass. |
| `result.tools[]._meta.fastmcp` | Implementation-specific metadata FastMCP attaches (e.g. version tags) — not part of the core MCP spec, safe to ignore for our purposes. |

## Frame 6 — `client -> server`: `tools/call` request

| Field | Meaning |
|---|---|
| `params.name` | Which tool to invoke — must be one of the names `tools/list` just returned. |
| `params.arguments` | The actual argument values the model chose to send, validated against that tool's `inputSchema` before the server's Python function ever runs. |

## Frame 7 — `server -> client`: `tools/call` response

| Field | Meaning |
|---|---|
| `result.content` | Human/model-readable content blocks (here, one `text` block containing the JSON-encoded return value) — this is what actually gets shown to the model as the tool's observation. |
| `result.isError` | `false` on success. On a genuinely recoverable failure (see `error_before_after.md`), this flips to `true` and `content` carries the exception's message text instead — no crash, no dropped connection, just a normal response the model can read and react to. |
| `result.structuredContent` | The same return value as a real JSON object (not a string) — used when a caller wants to parse the result programmatically instead of re-parsing the text block. |
