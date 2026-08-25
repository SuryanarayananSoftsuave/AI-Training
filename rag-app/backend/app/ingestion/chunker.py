from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass
class Chunk:
    text: str
    page_number: int
    section_heading: str | None
    content_type: str  # "text" | "table"


class _PlainSection:
    """Stand-in for a langchain Document when a page has no Markdown headers."""

    def __init__(self, text: str) -> None:
        self.page_content = text
        self.metadata: dict[str, str] = {}


def _isolate_tables(block: str) -> list[tuple[str, str]]:
    """Split a text block into ('text' | 'table', text) segments, pulling
    contiguous Markdown table line-runs out as their own atomic segments so
    the token splitter never cuts through a row — the single most common
    chunking failure mode on tabular PDFs.
    """
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    in_table = False

    def flush(kind: str) -> None:
        text = "\n".join(buf).strip()
        if text:
            segments.append((kind, text))
        buf.clear()

    for line in block.split("\n"):
        is_table_line = bool(_TABLE_LINE.match(line))
        if is_table_line != in_table:
            flush("table" if in_table else "text")
            in_table = is_table_line
        buf.append(line)
    flush("table" if in_table else "text")
    return segments


def _section_heading(metadata: dict[str, str]) -> str | None:
    for level in ("h3", "h2", "h1"):
        if level in metadata:
            return metadata[level]
    return None


def chunk_page(
    markdown: str,
    page_number: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    """Two-stage chunking: a structure-aware header split first (so section
    titles survive as metadata), then token-accurate size normalization on
    whatever remains — with Markdown tables isolated as atomic chunks.
    """
    if not markdown.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False)
    sections = header_splitter.split_text(markdown) or [_PlainSection(markdown)]

    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for section in sections:
        heading = _section_heading(section.metadata)
        for content_type, segment_text in _isolate_tables(section.page_content):
            if content_type == "table":
                chunks.append(Chunk(segment_text, page_number, heading, "table"))
            else:
                for piece in token_splitter.split_text(segment_text):
                    if piece.strip():
                        chunks.append(Chunk(piece, page_number, heading, "text"))
    return chunks
