"""Week 6 deliverable: deterministic checks that replace 2+ criteria
previously graded by the LLM judge.

Per the rubric's own "common mistakes" list: "Paying a model to check
whether a section reference resolves or whether the handbook version is
cited -- `if` and a lookup table do that for free and never have an off
day." These four checks are exactly that -- plain code against
`sample_data/hr_policy_kb_md/section_index.json` (ground truth) and the
`ChatStreamFinal`-shaped response a real `/chat` call already produces. No
LLM call, no prompt, nothing probabilistic -- same input always gives the
same output.

Each `assert_*` function is a pure, independently unit-testable predicate:
`(same-shaped args) -> (passed: bool, reason: str)`. `ASSERTIONS` is the
dispatch table `run_week6_eval.py` uses, keyed by the name a case's
`checks` list references, each wrapped to take the uniform
`(response: dict, case: dict) -> (bool, str)` shape a case-driven runner
needs.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

_SECTION_INDEX_PATH = (
    Path(__file__).resolve().parent.parent.parent / "sample_data" / "hr_policy_kb_md" / "section_index.json"
)
_SECTION_NUMBER_RE = re.compile(r"^[\s*_#]*(\d+(?:\.\d+)*)")
_VERSION_YEAR_RE = re.compile(r"-v(\d{4})$")
_NOTICE_PERIOD_RE = re.compile(r"\b(\d+)\s*(day|days|week|weeks)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _load_section_index() -> set[tuple[str, str]]:
    """(doc_slug, section_id) pairs, e.g. ("leave-policy-v2024", "3.1") --
    doc_slug matches a citation's filename with its extension stripped.
    """
    data = json.loads(_SECTION_INDEX_PATH.read_text(encoding="utf-8"))
    return {(s["doc"], s["section_id"]) for s in data["sections"]}


def assert_section_reference_valid(citations: list[dict]) -> tuple[bool, str]:
    """PASS iff at least one citation's (filename, section_heading) resolves
    to a real (doc, section_id) pair in the ground-truth index -- i.e. the
    answer is anchored to a real, numbered policy section, not just
    "somewhere in the document." Chunk `section_heading` metadata is the
    literal heading text (e.g. "3.1 Casual Leave", set by
    `ingestion/chunker.py`), so the leading number is extracted (tolerating
    a pymupdf4llm quirk where a bold PDF heading comes back Markdown-bolded,
    e.g. "**5. Carry-Over Rules**") and checked against the index rather
    than fuzzy-matching the title text.

    Known, real (not hypothetical) gap this surfaced: a `.docx` upload of
    the same policy converted from Word's auto-numbered list styling loses
    the number entirely in extracted text (python-docx's `paragraph.text`
    doesn't include list auto-numbers) -- e.g. a heading literally styled
    "3.1 Casual Leave" in Word extracts as just "Casual Leave". Since
    dedup is by exact file hash, a `.docx` and `.pdf` of the "same" policy
    both get indexed as separate documents, and if the `.docx` chunk wins
    reranking, this assertion correctly fails -- the citation genuinely
    isn't anchored to a numbered section in what was actually extracted.
    Not a bug in this function; a real content/ingestion limitation.
    """
    index = _load_section_index()
    if not citations:
        return False, "no citations to check"
    for c in citations:
        doc_slug = Path(c["filename"]).stem
        heading = (c.get("section_heading") or "").strip()
        match = _SECTION_NUMBER_RE.match(heading)
        if match and (doc_slug, match.group(1)) in index:
            return True, f"citation [{c['marker']}] resolves to {doc_slug} §{match.group(1)} ({heading!r})"
    return False, f"no citation's section_heading resolves to a real section -- headings seen: {[c.get('section_heading') for c in citations]}"


def assert_handbook_version_cited(answer_text: str, citations: list[dict]) -> tuple[bool, str]:
    """PASS iff every citation drawn from a VERSIONED policy doc (filename
    matches `*-v<year>.*`, e.g. leave-policy-v2023/v2024 -- the app's
    deliberate near-duplicate collision) has that version year spelled out
    somewhere in the answer text. Not applicable (vacuous PASS, noted in
    the reason) when no versioned doc was cited at all.
    """
    versioned = [c for c in citations if _VERSION_YEAR_RE.search(Path(c["filename"]).stem)]
    if not versioned:
        return True, "no versioned-policy citation in this answer -- assertion not applicable"
    for c in versioned:
        year = _VERSION_YEAR_RE.search(Path(c["filename"]).stem).group(1)
        if year not in answer_text:
            return False, f"citation [{c['marker']}] is from {Path(c['filename']).stem} but '{year}' never appears in the answer text"
    return True, "every versioned-policy citation's year is named in the answer text"


def assert_notice_period_numeric(answer_text: str) -> tuple[bool, str]:
    """PASS iff the answer names an actual number of days/weeks -- catches
    an answer that hedges in prose ("a reasonable notice period") instead
    of giving the figure the policy actually states.
    """
    match = _NOTICE_PERIOD_RE.search(answer_text)
    if match:
        return True, f"found numeric notice period: {match.group(0)!r}"
    return False, "no '<number> day(s)/week(s)' pattern found in the answer text"


def assert_out_of_jurisdiction_refused(verdict: str) -> tuple[bool, str]:
    """PASS iff the app's own off-topic gate fired (verdict ==
    'out_of_scope'). `notice-period-and-termination-policy.pdf` deliberately
    covers only India/US/UK, so a case asking about e.g. Germany or France
    is specifically chosen to exercise this gate.
    """
    if verdict == "out_of_scope":
        return True, "verdict is out_of_scope"
    return False, f"expected out_of_scope, got verdict={verdict!r}"


def _check_section_reference_valid(response: dict, case: dict) -> tuple[bool, str]:
    return assert_section_reference_valid(response["citations"])


def _check_handbook_version_cited(response: dict, case: dict) -> tuple[bool, str]:
    return assert_handbook_version_cited(response["answer"], response["citations"])


def _check_notice_period_numeric(response: dict, case: dict) -> tuple[bool, str]:
    return assert_notice_period_numeric(response["answer"])


def _check_out_of_jurisdiction_refused(response: dict, case: dict) -> tuple[bool, str]:
    verdict = response["judgment"]["verdict"] if response.get("judgment") else response.get("verdict", "")
    return assert_out_of_jurisdiction_refused(verdict)


# Dispatch table: case["checks"] names one or more of these; the runner
# calls each named check with the live response + the case dict. Keep this
# in sync with judge_v1.j2 -- each key added here is a criterion that
# should then be DELETED from the judge prompt (rubric requirement #2).
ASSERTIONS: dict[str, Callable[[dict, dict], tuple[bool, str]]] = {
    "section_reference_valid": _check_section_reference_valid,
    "handbook_version_cited": _check_handbook_version_cited,
    "notice_period_numeric": _check_notice_period_numeric,
    "out_of_jurisdiction_refused": _check_out_of_jurisdiction_refused,
}
