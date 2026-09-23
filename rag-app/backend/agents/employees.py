"""Mock employee records for Week 7 (W7-Task-Set-C.md) -- the rubric
assumes an existing employee-record system this app never had, so this is
a small, clearly-synthetic dataset invented for this exercise, not real
data.

`long_service_bonus_days` is this app's own additive HR rule (tenure >= 2
years earns +15 notice-period days on top of the jurisdiction base figure
from notice-period-and-termination-policy.pdf) -- it does not contradict
the handbook (which is flat-by-country, no tenure tiers); it's a
supplementary internal policy encoded in `get_notice_period_rule`
(agents/tools.py), invented specifically to give the race questions real
step-3-depends-on-step-2 branching, as the rubric requires.

Jurisdiction base notice-period figures below match what the real indexed
handbook actually states (confirmed via live retrieval during Week 6's
eval run): India=60, US=14, UK=30.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Jurisdiction = Literal["india", "us", "uk"]

_BASE_NOTICE_DAYS: dict[Jurisdiction, int] = {"india": 60, "us": 14, "uk": 30}
_LONG_SERVICE_BONUS_DAYS = 15
_LONG_SERVICE_THRESHOLD_YEARS = 2.0


@dataclass(frozen=True)
class Employee:
    employee_id: str
    name: str
    jurisdiction: Jurisdiction
    tenure_years: float


EMPLOYEES: dict[str, Employee] = {
    e.employee_id: e
    for e in [
        Employee("E1", "Asha Rao", "india", 5.0),
        Employee("E2", "Priya Nair", "india", 1.0),
        Employee("E3", "John Carter", "us", 3.0),
        Employee("E4", "Maria Gomez", "us", 0.5),
        Employee("E5", "Oliver Smith", "uk", 4.0),
        Employee("E6", "Emma Jones", "uk", 1.5),
        Employee("E7", "Vikram Singh", "india", 2.0),
        Employee("E8", "Lisa Wong", "us", 6.0),
        Employee("E9", "James Brown", "uk", 0.8),
        Employee("E10", "Deepa Iyer", "india", 10.0),
    ]
}


def expected_notice_days(jurisdiction: Jurisdiction, tenure_years: float) -> int:
    """Ground truth for the race harness's pass/fail check -- deterministic,
    same rule `get_notice_period_rule` (agents/tools.py) implements, kept
    here separately so the race harness's assertion doesn't just re-run the
    tool it's grading and call that a pass.
    """
    base = _BASE_NOTICE_DAYS[jurisdiction]
    bonus = _LONG_SERVICE_BONUS_DAYS if tenure_years >= _LONG_SERVICE_THRESHOLD_YEARS else 0
    return base + bonus
