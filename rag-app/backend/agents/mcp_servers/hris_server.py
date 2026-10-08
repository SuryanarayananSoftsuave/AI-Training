"""Week 9 (W9-Task-Set-C.md): the mock third-party server -- "People Ops has
stood up an MCP server over the HRIS." Deliberately a separate, independent
mock dataset from agents/employees.py (which backs the Week 7/8 tools) even
though it reuses the same E1-E10 ids for narrative continuity -- the point of
this exercise is bolting on a genuinely external system via config alone, not
another function living in our own app's data model.

Two plain lookups, no special error-recoverability requirement here (that
lives on OUR OWN server -- see policy_search_server.py's `as_of` handling and
WEEK9_MCP_RESEARCH.md's correction) -- an "unknown employee id" message is
good practice, not the graded mechanism.

Run standalone: `python -m agents.mcp_servers.hris_server`
"""
from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("hris")

_HRIS_RECORDS: dict[str, dict] = {
    "E1": {"grade_band": "G6", "accrued_leave_days": 18.5},
    "E2": {"grade_band": "G3", "accrued_leave_days": 6.0},
    "E3": {"grade_band": "G5", "accrued_leave_days": 11.0},
    "E4": {"grade_band": "G2", "accrued_leave_days": 3.5},
    "E5": {"grade_band": "G5", "accrued_leave_days": 14.0},
    "E6": {"grade_band": "G3", "accrued_leave_days": 8.0},
    "E7": {"grade_band": "G4", "accrued_leave_days": 9.5},
    "E8": {"grade_band": "G7", "accrued_leave_days": 22.0},
    "E9": {"grade_band": "G2", "accrued_leave_days": 4.5},
    "E10": {"grade_band": "G8", "accrued_leave_days": 30.0},
}


@mcp.tool()
def get_grade_band(employee_id: str) -> dict:
    """Look up an employee's HRIS grade band by employee id."""
    record = _HRIS_RECORDS.get(employee_id)
    if record is None:
        raise ValueError(f"no HRIS record for employee_id {employee_id!r}")
    return {"employee_id": employee_id, "grade_band": record["grade_band"]}


@mcp.tool()
def get_accrued_leave_balance(employee_id: str) -> dict:
    """Look up an employee's current accrued leave balance, in days, from the HRIS."""
    record = _HRIS_RECORDS.get(employee_id)
    if record is None:
        raise ValueError(f"no HRIS record for employee_id {employee_id!r}")
    return {"employee_id": employee_id, "accrued_leave_days": record["accrued_leave_days"]}


if __name__ == "__main__":
    mcp.run(transport="stdio")
