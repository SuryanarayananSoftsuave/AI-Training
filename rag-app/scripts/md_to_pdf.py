"""Converts Markdown knowledge-base articles to PDF, preserving headings,
tables, and lists, so they can be uploaded through the RAG app's PDF-only
ingestion pipeline without any backend changes.

Usage:
    python md_to_pdf.py <input_dir_or_file> [output_dir]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_HEADING_STYLES = {1: "Heading1", 2: "Heading2", 3: "Heading3", 4: "Heading4"}

_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_LIST_ITEM_RE = re.compile(r"^([-*]|\d+\.)\s+(.*)$")

# ReportLab's base PDF fonts (Helvetica etc.) only cover WinAnsiEncoding, not
# full Unicode -- characters outside it render as a blank/box glyph rather
# than raising an error. Worse, even WinAnsi-covered punctuation (en dash,
# curly quotes) that *renders* fine on screen comes back as U+FFFD when a
# downstream parser (pymupdf4llm) extracts text from the PDF, because
# ReportLab doesn't emit a ToUnicode CMap for the base14 fonts -- silently
# corrupting the very text that gets chunked and embedded. Since this PDF
# only exists to survive a round trip through a PDF text extractor, every
# non-ASCII typographic character is normalized to a plain-ASCII equivalent
# up front rather than preserved for appearance. Keys are \\uXXXX escapes,
# not literal characters, so the mapping survives any editor/encoding.
_UNICODE_NORMALIZE = {
    "‐": "-",    # hyphen
    "‑": "-",    # non-breaking hyphen
    "‒": "-",    # figure dash
    "–": "-",    # en dash
    "—": "--",   # em dash
    "→": "->",   # rightwards arrow
    "‘": "'",    # left single quotation mark
    "’": "'",    # right single quotation mark
    "“": '"',    # left double quotation mark
    "”": '"',    # right double quotation mark
    "·": "-",    # middle dot
    " ": " ",    # no-break space
    " ": " ",    # narrow no-break space
}


def _normalize_unicode(text: str) -> str:
    for src, dst in _UNICODE_NORMALIZE.items():
        text = text.replace(src, dst)
    return text


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_markdown_to_reportlab(text: str) -> str:
    """Markdown inline spans -> reportlab mini-markup.

    Code spans are stashed behind placeholder tokens before bold/italic
    matching runs, and restored last -- otherwise a stray `*` inside a
    code span (e.g. `` `SHR-*` ``) gets misread as an italic delimiter
    and produces mismatched/overlapping tags.
    """
    text = _escape(text)
    code_spans: list[str] = []

    def _stash_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"\x00{len(code_spans) - 1}\x00"

    text = _CODE_RE.sub(_stash_code, text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _CODE_PLACEHOLDER_RE.sub(lambda m: f'<font face="Courier">{code_spans[int(m.group(1))]}</font>', text)
    return text


def _strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[end + 4:].lstrip("\n")


def _parse_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        if re.match(r"^\s*\|?\s*[-: ]+\|[-:| ]*\s*$", line):
            continue  # header separator row, e.g. |---|---|
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def convert(md_path: Path, pdf_path: Path) -> None:
    raw = _normalize_unicode(md_path.read_text(encoding="utf-8"))
    meta, body = _strip_frontmatter(raw)

    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    story = []

    story.append(Paragraph(_inline_markdown_to_reportlab(meta.get("title", md_path.stem)), styles["Title"]))
    meta_line = " | ".join(f"{k}: {v}" for k, v in meta.items() if k != "title")
    if meta_line:
        story.append(Paragraph(_escape(meta_line), styles["Italic"]))
    story.append(Spacer(1, 12))

    lines = body.splitlines()
    i, n = 0, len(lines)

    while i < n:
        stripped = lines[i].strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            story.append(Paragraph(_inline_markdown_to_reportlab(heading_match.group(2)), styles[_HEADING_STYLES.get(level, "Heading4")]))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = _parse_table_rows(table_lines)
            if rows:
                wrapped = [[Paragraph(_inline_markdown_to_reportlab(cell), body_style) for cell in row] for row in rows]
                table = Table(wrapped, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E6D62")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]))
                story.append(table)
                story.append(Spacer(1, 10))
            continue

        list_match = _LIST_ITEM_RE.match(stripped)
        if list_match:
            items = []
            while i < n:
                s = lines[i].strip()
                m = _LIST_ITEM_RE.match(s)
                if m:
                    items.append(m.group(2))
                    i += 1
                elif s and not _HEADING_RE.match(s) and not s.startswith("|"):
                    if items:
                        items[-1] += " " + s  # wrapped continuation of the previous item
                    i += 1
                else:
                    break
            for item_text in items:
                story.append(Paragraph("&bull; " + _inline_markdown_to_reportlab(item_text), body_style))
            story.append(Spacer(1, 6))
            continue

        # Plain paragraph: absorb following lines until a blank line or a new block starts.
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not (_HEADING_RE.match(lines[i].strip()) or _LIST_ITEM_RE.match(lines[i].strip()) or lines[i].strip().startswith("|")):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(_inline_markdown_to_reportlab(" ".join(para_lines)), body_style))
        story.append(Spacer(1, 6))

    SimpleDocTemplate(str(pdf_path), pagesize=LETTER).build(story)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python md_to_pdf.py <input_dir_or_file> [output_dir]")
        raise SystemExit(1)

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = [src] if src.is_file() else sorted(src.glob("*.md"))
    if not md_files:
        print(f"no .md files found in {src}")
        return

    for md_file in md_files:
        pdf_file = out_dir / (md_file.stem + ".pdf")
        convert(md_file, pdf_file)
        print(f"wrote {pdf_file}")


if __name__ == "__main__":
    main()
