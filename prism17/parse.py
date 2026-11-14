"""Page-level PDF parsing for Prism17.

The parser deliberately keeps two things separate:

1. Raw page text, used later by quote verification.
2. Passage-level text, used for retrieval and judging.

Tables are extracted with pdfplumber and stored as passages with
``is_table=True``. If a page cannot be read (for example, a scanned image
without OCR), it is recorded as an unreadable page instead of crashing the
document.

Run:
    python -m prism17.parse data/reports/example.pdf
    python -m prism17.parse data/reports/example.pdf --out outputs/example.parse.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # PyMuPDF >= 1.24 exposes the non-deprecated ``pymupdf`` name.
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF
    import fitz  # type: ignore[no-redef]

from .config import display_path, get, load_config, resolve_path

LOGGER = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“])")
_TRAILING_PUNCTUATION = (".", "!", "?", ":", ";")
_HEADING_TERMINAL_PUNCTUATION = (".", "!", "?")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    """A retrievable unit of report text."""

    passage_id: str
    page: int
    section_heading: str | None
    text: str
    is_table: bool = False
    table_index: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["word_count"] = self.word_count
        return data


@dataclass
class PageText:
    """Raw text for one PDF page, retained for quote verification."""

    page: int
    text: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnreadablePage:
    """A page with no usable extractable text."""

    page: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    """Everything the rest of the pipeline needs from one PDF."""

    pdf_path: str
    title: str
    page_count: int
    passages: list[Passage] = field(default_factory=list)
    pages: list[PageText] = field(default_factory=list)
    unreadable_pages: list[UnreadablePage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    parse_seconds: float = 0.0

    @property
    def table_count(self) -> int:
        return sum(1 for passage in self.passages if passage.is_table)

    def page_text(self, page_number: int) -> str | None:
        """Return raw page text for a 1-indexed page number."""
        for page in self.pages:
            if page.page == page_number:
                return page.text
        return None

    def to_dict(self, include_pages: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": "1.0",
            "document": {
                "pdf_path": self.pdf_path,
                "title": self.title,
                "page_count": self.page_count,
                "sha256": self.metadata.get("sha256"),
                "metadata": self.metadata,
            },
            "stats": self.stats,
            "parse_seconds": round(self.parse_seconds, 3),
            "passages": [passage.to_dict() for passage in self.passages],
            "unreadable_pages": [page.to_dict() for page in self.unreadable_pages],
            "warnings": self.warnings,
        }
        if include_pages:
            data["pages"] = [page.to_dict() for page in self.pages]
        return data


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and trim."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def word_count(text: str) -> int:
    """Count words using whitespace splitting."""
    return len(normalise_whitespace(text).split())


def split_text_to_passages(text: str, max_words: int = 250) -> list[str]:
    """Split one long block into sentence-aware chunks of at most max_words.

    A single sentence longer than max_words is split on word boundaries as a
    fallback. Empty chunks are dropped.
    """
    text = normalise_whitespace(text)
    if not text:
        return []
    if max_words < 10:
        raise ValueError("max_words must be at least 10")

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)
        if sentence_words > max_words:
            # Flush anything pending, then split the oversized sentence.
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
            words = sentence.split()
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start : start + max_words]).strip())
            continue

        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _clean_block_text(text: str) -> str:
    """Normalise a PDF block while keeping sentence punctuation."""
    return normalise_whitespace(text)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


@dataclass
class _RawBlock:
    text: str
    bbox: tuple[float, float, float, float]
    max_font_size: float
    is_bold: bool
    is_heading: bool = False


def _span_is_bold(span: dict[str, Any]) -> bool:
    """PyMuPDF uses bit 4 (value 16) for bold text in span flags."""
    return bool(int(span.get("flags", 0)) & 16)


def _extract_page_blocks(page: fitz.Page) -> tuple[list[_RawBlock], float]:
    """Extract text blocks plus a best-effort body font size.

    Heading detection is heuristic. It is intentionally conservative: a missed
    heading is harmless, while turning body text into a heading would drop
    content from the passage stream.
    """
    data = page.get_text("dict", sort=True)
    blocks: list[_RawBlock] = []
    weighted_sizes: list[float] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        lines: list[str] = []
        block_sizes: list[float] = []
        block_is_bold = False

        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", ""))
                if not span_text.strip():
                    continue
                line_text_parts.append(span_text)
                size = float(span.get("size", 0.0))
                block_sizes.append(size)
                # Weight by character count so a single large label does not
                # dominate the body-size estimate.
                weighted_sizes.extend([size] * max(1, len(span_text)))
                block_is_bold = block_is_bold or _span_is_bold(span)
            line_text = "".join(line_text_parts).strip()
            if line_text:
                lines.append(line_text)

        text = _clean_block_text(" ".join(lines))
        if not text:
            continue

        bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        bbox = tuple(float(value) for value in bbox_raw[:4])
        max_size = max(block_sizes) if block_sizes else 0.0
        blocks.append(
            _RawBlock(
                text=text,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                max_font_size=max_size,
                is_bold=block_is_bold,
            )
        )

    body_size = statistics.median(weighted_sizes) if weighted_sizes else 10.0
    for block in blocks:
        block.is_heading = _looks_like_heading(block, body_size)
    return blocks, body_size


def _looks_like_heading(block: _RawBlock, body_size: float) -> bool:
    """Best-effort heading detector based on size, boldness and length."""
    text = block.text.strip()
    words = text.split()
    if len(words) < 1 or len(words) > 18:
        return False
    if len(text) < 3 or text.endswith(_HEADING_TERMINAL_PUNCTUATION):
        return False
    if any(ord(char) < 32 for char in text):
        return False
    if text.upper().replace(" ", "") in {"NIL", "NILNIL"}:
        return False
    if block.max_font_size >= body_size * 1.15:
        return True
    if block.is_bold and len(words) <= 12 and block.max_font_size >= body_size * 0.95:
        return True
    # Short all-caps labels are common section headers in BRSR filings even
    # when the PDF does not mark them bold.
    if text.isupper() and len(words) <= 12:
        return True
    return False


def _bbox_overlap_ratio(
    block_bbox: Sequence[float], table_bbox: Sequence[float]
) -> float:
    """Return the fraction of a block's area that falls inside a table bbox."""
    bx0, by0, bx1, by1 = (float(v) for v in block_bbox[:4])
    tx0, ty0, tx1, ty1 = (float(v) for v in table_bbox[:4])
    block_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if block_area <= 0:
        return 0.0
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return intersection / block_area


def _table_to_text(rows: Iterable[Sequence[Any]]) -> str:
    """Render extracted table rows as a pipe-separated text passage."""
    rendered: list[str] = []
    for row in rows:
        cells = [normalise_whitespace(str(cell or "")) for cell in row]
        if any(cells):
            rendered.append(" | ".join(cells).strip())
    return "\n".join(rendered)


def extract_tables(
    pdf_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    """Extract tables page by page with pdfplumber.

    Returns:
        A mapping of 1-indexed page number to table records, plus warnings.
        Table records contain ``text``, ``rows``, ``cols`` and ``bbox``.
    """
    cfg = config or {}
    if not bool(get(cfg, "parsing", "extract_tables", default=True)):
        return {}, []

    min_rows = int(get(cfg, "parsing", "table_min_rows", default=2))
    min_cols = int(get(cfg, "parsing", "table_min_cols", default=2))
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    try:
        import pdfplumber
    except ImportError:
        LOGGER.warning("pdfplumber is not installed; table extraction skipped")
        return {}, ["pdfplumber not installed; table extraction skipped"]

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    found_tables = page.find_tables()
                except Exception as exc:  # noqa: BLE001 - a bad page must not stop parsing
                    warning = f"Table detection failed on page {page_number}: {exc}"
                    LOGGER.warning(warning)
                    warnings.append(warning)
                    continue

                for table_index, table in enumerate(found_tables, start=1):
                    try:
                        rows = table.extract()
                    except Exception as exc:  # noqa: BLE001
                        warning = (
                            f"Table extraction failed on page {page_number}, "
                            f"table {table_index}: {exc}"
                        )
                        LOGGER.warning(warning)
                        warnings.append(warning)
                        continue

                    if not rows:
                        continue
                    cleaned_rows = [
                        [normalise_whitespace(str(cell or "")) for cell in row]
                        for row in rows
                    ]
                    cleaned_rows = [row for row in cleaned_rows if any(row)]
                    if len(cleaned_rows) < min_rows:
                        continue
                    col_count = max((len(row) for row in cleaned_rows), default=0)
                    if col_count < min_cols:
                        continue

                    text = _table_to_text(cleaned_rows)
                    if word_count(text) < 3:
                        continue

                    bbox_raw = getattr(table, "bbox", None)
                    bbox = (
                        tuple(float(v) for v in bbox_raw[:4])
                        if bbox_raw is not None
                        else None
                    )
                    tables_by_page.setdefault(page_number, []).append(
                        {
                            "text": text,
                            "rows": len(cleaned_rows),
                            "cols": col_count,
                            "bbox": bbox,
                        }
                    )
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are reported, not fatal
        warning = f"pdfplumber could not open or read the PDF: {exc}"
        LOGGER.warning(warning)
        warnings.append(warning)

    return tables_by_page, warnings


# ---------------------------------------------------------------------------
# Passage construction
# ---------------------------------------------------------------------------


def _pack_blocks_into_passages(
    blocks: list[_RawBlock],
    page_number: int,
    section_heading: str | None,
    min_words: int,
    max_words: int,
    passage_counter: list[int],
) -> list[Passage]:
    """Merge adjacent blocks into passage-sized chunks.

    ``passage_counter`` is a one-element mutable counter shared across a page so
    passage IDs are stable and unique.
    """
    passages: list[Passage] = []
    buffer: list[str] = []

    def next_id() -> str:
        passage_counter[0] += 1
        return f"p{page_number:04d}-{passage_counter[0]:03d}"

    def emit(text: str) -> None:
        cleaned = normalise_whitespace(text)
        # Ignore page furniture such as bare page numbers. Very short blocks
        # are not useful retrieval units and are usually headers or footers.
        if word_count(cleaned) < 3:
            return
        for chunk in split_text_to_passages(cleaned, max_words=max_words):
            passages.append(
                Passage(
                    passage_id=next_id(),
                    page=page_number,
                    section_heading=section_heading,
                    text=chunk,
                    is_table=False,
                )
            )

    def flush() -> None:
        nonlocal buffer
        if buffer:
            emit(" ".join(buffer))
            buffer = []

    for block in blocks:
        text = _clean_block_text(block.text)
        if not text:
            continue
        if re.fullmatch(r"[\d\s./%()\-]+", text):
            # Bare numbers are page numbers, table-of-contents leaders, etc.
            continue

        block_words = word_count(text)
        if block_words >= max_words:
            flush()
            emit(text)
            continue

        if buffer and word_count(" ".join(buffer)) + block_words > max_words:
            flush()

        buffer.append(text)

        # Flush once a block ends a sentence and the buffer has enough content.
        # This keeps passages paragraph-like rather than page-long.
        if (
            word_count(" ".join(buffer)) >= min_words
            and text.endswith(_TRAILING_PUNCTUATION)
        ):
            flush()

    flush()
    return passages


def _make_table_passages(
    page_number: int,
    tables: list[dict[str, Any]],
    section_heading: str | None,
    passage_counter: list[int],
) -> list[Passage]:
    """Turn extracted table records into passages flagged as tables."""
    passages: list[Passage] = []
    for table_index, table in enumerate(tables, start=1):
        text = normalise_whitespace(str(table.get("text", "")))
        if not text:
            continue
        passage_counter[0] += 1
        passages.append(
            Passage(
                passage_id=f"p{page_number:04d}-t{passage_counter[0]:03d}",
                page=page_number,
                section_heading=section_heading,
                text=text,
                is_table=True,
                table_index=table_index,
                bbox=table.get("bbox"),
            )
        )
    return passages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(
    pdf_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    extract_tables_enabled: bool | None = None,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Parse a PDF into pages and passages.

    Args:
        pdf_path: Path to the PDF.
        config: Loaded config mapping. Defaults to config.yaml.
        extract_tables_enabled: Override the config table switch.
        max_pages: Optional development limit; page_count still reflects the
            pages actually parsed.
    """
    started = time.perf_counter()
    cfg = config or load_config()
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    min_words = int(get(cfg, "parsing", "min_passage_words", default=50))
    max_words = int(get(cfg, "parsing", "max_passage_words", default=250))
    unreadable_chars = int(
        get(cfg, "parsing", "unreadable_char_threshold", default=20)
    )
    if min_words < 1:
        raise ValueError("parsing.min_passage_words must be positive")
    if max_words < min_words:
        raise ValueError("parsing.max_passage_words must be >= min_passage_words")

    tables_enabled = (
        bool(get(cfg, "parsing", "extract_tables", default=True))
        if extract_tables_enabled is None
        else extract_tables_enabled
    )

    warnings: list[str] = []
    if tables_enabled:
        tables_by_page, table_warnings = extract_tables(path, cfg)
        warnings.extend(table_warnings)
    else:
        tables_by_page = {}

    try:
        document = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001 - report a clear parse error
        raise ValueError(f"Could not open PDF {path.name}: {exc}") from exc

    passages: list[Passage] = []
    pages: list[PageText] = []
    unreadable_pages: list[UnreadablePage] = []
    current_heading: str | None = None
    page_count = document.page_count
    pages_to_parse = min(page_count, max_pages) if max_pages else page_count

    try:
        for page_index in range(pages_to_parse):
            page_number = page_index + 1
            try:
                page = document.load_page(page_index)
                raw_page_text = page.get_text("text", sort=True) or ""
            except Exception as exc:  # noqa: BLE001
                warning = f"Page {page_number} could not be read: {exc}"
                LOGGER.warning(warning)
                warnings.append(warning)
                unreadable_pages.append(
                    UnreadablePage(page=page_number, reason=str(exc))
                )
                continue

            cleaned_page_text = normalise_whitespace(raw_page_text)
            pages.append(
                PageText(
                    page=page_number,
                    text=raw_page_text,
                    char_count=len(cleaned_page_text),
                )
            )

            page_tables = tables_by_page.get(page_number, [])
            if len(cleaned_page_text) < unreadable_chars and not page_tables:
                reason = "no extractable text (possibly a scanned page)"
                unreadable_pages.append(UnreadablePage(page=page_number, reason=reason))
                LOGGER.info("Page %s is unreadable: %s", page_number, reason)
                continue
            if len(cleaned_page_text) < unreadable_chars and page_tables:
                warnings.append(
                    f"Page {page_number} has little body text but contains tables"
                )

            try:
                raw_blocks, _body_size = _extract_page_blocks(page)
            except Exception as exc:  # noqa: BLE001
                warning = f"Block extraction failed on page {page_number}: {exc}"
                LOGGER.warning(warning)
                warnings.append(warning)
                raw_blocks = []

            table_bboxes = [
                table["bbox"] for table in page_tables if table.get("bbox") is not None
            ]

            def is_in_table(block: _RawBlock) -> bool:
                return any(
                    _bbox_overlap_ratio(block.bbox, table_bbox) >= 0.35
                    for table_bbox in table_bboxes
                )

            body_blocks: list[_RawBlock] = []
            page_counter = [0]
            for block in raw_blocks:
                if block.is_heading:
                    # Flush the current section before switching headings.
                    if body_blocks:
                        passages.extend(
                            _pack_blocks_into_passages(
                                body_blocks,
                                page_number,
                                current_heading,
                                min_words,
                                max_words,
                                page_counter,
                            )
                        )
                        body_blocks = []
                    current_heading = block.text
                    continue
                if is_in_table(block):
                    continue
                body_blocks.append(block)

            if body_blocks:
                passages.extend(
                    _pack_blocks_into_passages(
                        body_blocks,
                        page_number,
                        current_heading,
                        min_words,
                        max_words,
                        page_counter,
                    )
                )

            if page_tables:
                passages.extend(
                    _make_table_passages(
                        page_number,
                        page_tables,
                        current_heading,
                        page_counter,
                    )
                )
    finally:
        document.close()

    # Running headers and repeated table headers are often detected as
    # headings because they are bold. If a "heading" recurs on several pages,
    # treat it as page furniture and clear it rather than mislabelling whole
    # sections.
    _demote_repeated_headings(passages)
    _merge_short_passages(passages, min_words, max_words)

    body_passages = sum(1 for passage in passages if not passage.is_table)
    table_passages = sum(1 for passage in passages if passage.is_table)
    total_words = sum(passage.word_count for passage in passages)
    elapsed = time.perf_counter() - started

    metadata = {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "file_name": path.name,
    }

    stats = {
        "pages_parsed": pages_to_parse,
        "page_count": page_count,
        "passage_count": len(passages),
        "body_passage_count": body_passages,
        "table_passage_count": table_passages,
        "unreadable_page_count": len(unreadable_pages),
        "total_passage_words": total_words,
    }

    LOGGER.info(
        "Parsed %s pages, %s passages (%s tables), %s unreadable pages in %.2fs",
        pages_to_parse,
        len(passages),
        table_passages,
        len(unreadable_pages),
        elapsed,
    )

    return ParsedDocument(
        pdf_path=display_path(path, cfg),
        title=path.stem,
        page_count=pages_to_parse,
        passages=passages,
        pages=pages,
        unreadable_pages=unreadable_pages,
        warnings=warnings,
        metadata=metadata,
        stats=stats,
        parse_seconds=elapsed,
    )


def _demote_repeated_headings(
    passages: list[Passage], min_distinct_pages: int = 3
) -> None:
    """Clear headings that recur across several pages.

    A repeated bold line is much more likely to be a running header or a
    repeated table header than a genuine section title. Clearing it is safer
    than assigning the same false heading to many passages.
    """
    pages_by_heading: dict[str, set[int]] = {}
    for passage in passages:
        heading = passage.section_heading
        if heading:
            pages_by_heading.setdefault(heading, set()).add(passage.page)

    repeated = {
        heading
        for heading, pages in pages_by_heading.items()
        if len(pages) >= min_distinct_pages
    }
    if not repeated:
        return

    for passage in passages:
        if passage.section_heading in repeated:
            passage.section_heading = None


def _merge_short_passages(
    passages: list[Passage], min_words: int, max_words: int
) -> None:
    """Merge short adjacent body passages on the same page and section.

    Page and section boundaries are never crossed, so every merged passage
    still has an unambiguous page number. Table passages are left untouched.
    """
    merged: list[Passage] = []
    for passage in passages:
        if not merged:
            merged.append(passage)
            continue

        previous = merged[-1]
        can_merge = (
            not previous.is_table
            and not passage.is_table
            and previous.page == passage.page
            and previous.section_heading == passage.section_heading
            and (
                previous.word_count < min_words
                or passage.word_count < min_words
            )
            and previous.word_count + passage.word_count <= max_words
        )
        if can_merge:
            previous.text = f"{previous.text} {passage.text}".strip()
            continue

        merged.append(passage)

    passages[:] = merged


def _sha256(path: Path) -> str:
    """Hash a file in chunks for stable document identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(document: ParsedDocument) -> str:
    """Return a short human-readable parse summary."""
    stats = document.stats
    lines = [
        f"Document: {document.title}",
        f"Pages parsed: {stats.get('pages_parsed')}",
        f"Passages: {stats.get('passage_count')} "
        f"(body {stats.get('body_passage_count')}, "
        f"tables {stats.get('table_passage_count')})",
        f"Unreadable pages: {stats.get('unreadable_page_count')}",
        f"Parse time: {document.parse_seconds:.2f}s",
    ]
    if document.unreadable_pages:
        page_list = ", ".join(str(page.page) for page in document.unreadable_pages)
        lines.append(f"Unreadable page numbers: {page_list}")
    if document.warnings:
        lines.append(f"Warnings: {len(document.warnings)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Phase 1 parsing."""
    parser = argparse.ArgumentParser(
        description="Parse a PDF into page-level passages for Prism17."
    )
    parser.add_argument("pdf", help="Path to the PDF report")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write full parse JSON here")
    parser.add_argument(
        "--no-tables", action="store_true", help="Skip pdfplumber table extraction"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Parse only the first N pages"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of passage previews to print (default: 3)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        document = parse_pdf(
            args.pdf,
            config,
            extract_tables_enabled=not args.no_tables,
            max_pages=args.max_pages,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(summarise(document))
    if document.warnings:
        print("\nWarnings:")
        for warning in document.warnings[:10]:
            print(f"  - {warning}")
        if len(document.warnings) > 10:
            print(f"  ... {len(document.warnings) - 10} more")

    if args.preview > 0 and document.passages:
        print(f"\nFirst {min(args.preview, len(document.passages))} passage(s):")
        for passage in document.passages[: args.preview]:
            label = "table" if passage.is_table else "body"
            heading = passage.section_heading or "unlabelled section"
            print(
                f"\n[{passage.passage_id}] page {passage.page} | {label} | "
                f"{heading} | {passage.word_count} words"
            )
            preview = passage.text[:400]
            print(preview + ("..." if len(passage.text) > 400 else ""))

    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = resolve_path(out_path, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote parse JSON: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
