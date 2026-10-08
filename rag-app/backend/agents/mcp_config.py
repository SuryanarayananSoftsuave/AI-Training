"""Loads the MCP server config agents/mcp_agent.py connects with.

The two JSON files on disk (mcp_config_one_server.json / _two_servers.json)
intentionally omit the interpreter path -- on Windows, a bare "python" in the
config risks resolving to whatever python is first on PATH, not necessarily
this project's own backend/.venv (the documented dev workflow runs uvicorn
via `.venv\\Scripts\\python.exe` directly, which does not add that venv's
Scripts dir to PATH for child processes). This loader injects `sys.executable`
(the interpreter actually running this process) and an explicit `cwd` (this
backend/ directory, so `agents.*`/`app.*` imports resolve inside the spawned
subprocess the same way they do for the main FastAPI process) at connect
time instead of baking either into the checked-in config files.

Diffing the two JSON files directly IS the required "config diff" deliverable
for W9-Task-Set-C.md -- see mcp_diff.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATHS: dict[str, Path] = {
    "one": _BACKEND_DIR / "agents" / "mcp_config_one_server.json",
    "two": _BACKEND_DIR / "agents" / "mcp_config_two_servers.json",
}


def config_path(label: Literal["one", "two"]) -> Path:
    return _CONFIG_PATHS[label]


def load_config(label: Literal["one", "two"]) -> dict:
    with config_path(label).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    servers = {
        name: {"command": sys.executable, "cwd": str(_BACKEND_DIR), **spec}
        for name, spec in raw["mcpServers"].items()
    }
    return {"mcpServers": servers}
