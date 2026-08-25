from __future__ import annotations

from dataclasses import dataclass

import pymupdf4llm


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    markdown: str


def parse_pdf(path: str) -> list[ParsedPage]:
    """Parse a PDF into per-page Markdown, preserving headings and rendering
    tables as Markdown tables so the chunker can reason about structure
    instead of raw, layout-stripped text.
    """
    pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    return [ParsedPage(page_number=i + 1, markdown=page["text"]) for i, page in enumerate(pages)]
