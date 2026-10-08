"""Week 9 (W9-Task-Set-C.md): produces real `git diff` output between two
plain files without needing either one staged or committed -- `git diff
--no-index` works on any two paths regardless of git tracking state, which
matters here since this project's standing convention is never to commit
without being explicitly asked. Used for both the required agent_diff.txt
(mcp_agent.py vs. its frozen baseline -- must always be empty) and the
config diff (one-server vs. two-server JSON -- expected to show only the
added "hris" block).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent


def diff_files(path_a: Path, path_b: Path) -> tuple[str, bool]:
    """`git diff --no-index` exits 1 when the files differ and 0 when
    they're identical -- neither is a real error, so this never raises on
    either outcome.
    """
    result = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", "--", str(path_a), str(path_b)],
        capture_output=True, text=True, check=False,
    )
    diff_text = result.stdout.strip()
    return diff_text, diff_text == ""


def agent_diff() -> tuple[str, bool]:
    return diff_files(_AGENTS_DIR / "mcp_agent_baseline.py", _AGENTS_DIR / "mcp_agent.py")


def config_diff() -> tuple[str, bool]:
    return diff_files(_AGENTS_DIR / "mcp_config_one_server.json", _AGENTS_DIR / "mcp_config_two_servers.json")
