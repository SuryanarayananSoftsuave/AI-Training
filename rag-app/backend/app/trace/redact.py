from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_EMPLOYEE_ID_RE = re.compile(r"\bEMP-?\d{4,}\b", re.IGNORECASE)


def redact(text: str) -> str:
    """Scrubs patterns that would be real PII in a production deployment.
    This app's KB is synthetic HR content, not real employee records, but a
    question typed by a real user could still name a real email, phone
    number, or employee ID -- applied to `question`/`answer` at the point a
    trace is constructed, before it's ever written to disk, never as a
    later pass over already-written data.
    """
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _EMPLOYEE_ID_RE.sub("[redacted-employee-id]", text)
    return text
