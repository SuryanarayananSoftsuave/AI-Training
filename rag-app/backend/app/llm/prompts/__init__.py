from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# autoescape=False is explicit, not just the default -- these templates feed
# an LLM prompt, not HTML. HR policy text pulled from PDFs routinely contains
# "&" (e.g. "Health & Insurance") and "<"/">" characters; autoescaping would
# corrupt that text into "&amp;"/"&lt;" before it ever reaches the model.
_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=False,
    trim_blocks=True,
)

# Templates are never edited in place -- a prompt change ships as a new
# versioned file (e.g. judge_v2.j2) so a trace recorded against judge_v1 can
# always be replayed against the exact historical prompt, even after v2 ships.
# generation_v2 exists because the generation task itself changed shape (v1
# closed with "Respond with the answer text and the list of source markers
# you actually used", written for a structured-JSON response with a separate
# indices field; streaming has no structured field, so v2 asks for plain
# text with inline citations only -- v1 is kept for replaying old traces.
# generation_v3 exists because gpt-oss (via Groq), found via real live usage,
# ignores v2's example and cites in its own trained "[N†...]" full-width-
# bracket style instead of plain "[1]" -- v3 explicitly spells out the exact
# required format and explicitly forbids that style. `chat_service.py`'s
# citation regex was ALSO broadened to recognize gpt-oss's style regardless
# (defense in depth -- prompt instructions alone weren't reliable enough to
# fully suppress a model's own trained citation habit).
_LIVE_VERSIONS = {
    "query_expansion": "v1",
    "generation": "v3",
    "judge": "v1",
}


def get_live_versions() -> dict[str, str]:
    """Which template version is currently live for each prompt task --
    recorded into every chat trace so replay always re-renders the exact
    historical prompt, even after a later version ships.
    """
    return dict(_LIVE_VERSIONS)


def render_query_expansion(question: str, variant_count: int, version: str | None = None) -> str:
    version = version or _LIVE_VERSIONS["query_expansion"]
    template = _ENV.get_template(f"query_expansion_{version}.j2")
    return template.render(question=question, variant_count=variant_count)


def render_generation(question: str, numbered_context: str, version: str | None = None) -> str:
    version = version or _LIVE_VERSIONS["generation"]
    template = _ENV.get_template(f"generation_{version}.j2")
    return template.render(question=question, numbered_context=numbered_context)


def render_judge(question: str, answer: str, numbered_context: str, version: str | None = None) -> str:
    version = version or _LIVE_VERSIONS["judge"]
    template = _ENV.get_template(f"judge_{version}.j2")
    return template.render(question=question, answer=answer, numbered_context=numbered_context)
