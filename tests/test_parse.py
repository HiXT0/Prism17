"""Tests for page-level parsing and passage construction."""

from __future__ import annotations

from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

import pytest

from prism17.parse import (
    Passage,
    _bbox_overlap_ratio,
    _merge_short_passages,
    _table_to_text,
    normalise_whitespace,
    parse_pdf,
    split_text_to_passages,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "reports" / "sjs_india_brsr_fy2023_24.pdf"


def _new_pdf(path: Path) -> tuple[fitz.Document, fitz.Page]:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    return document, page


def test_normalise_whitespace() -> None:
    assert normalise_whitespace("  a\n\tb   c  ") == "a b c"


def test_split_text_respects_max_words() -> None:
    text = " ".join(f"word{i}" for i in range(600)) + "."
    chunks = split_text_to_passages(text, max_words=100)
    assert len(chunks) == 6
    assert all(len(chunk.split()) <= 100 for chunk in chunks)
    assert " ".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_text_keeps_sentences_together_when_possible() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_text_to_passages(text, max_words=50)
    assert chunks == [text]


def test_split_text_rejects_tiny_max_words() -> None:
    with pytest.raises(ValueError):
        split_text_to_passages("some text", max_words=5)


def test_table_to_text_drops_empty_cells() -> None:
    text = _table_to_text(
        [
            ["Metric", "FY2023", "FY2024"],
            ["Scope 1 emissions", "", "1,200 tCO2e"],
            [None, None, None],
        ]
    )
    assert "Metric | FY2023 | FY2024" in text
    assert "Scope 1 emissions |  | 1,200 tCO2e" in text
    assert text.count("\n") == 1


def test_short_passages_merge_only_within_page_and_section() -> None:
    passages = [
        Passage("p1", 1, "Section A", "short one"),
        Passage("p2", 1, "Section A", "short two"),
        Passage("p3", 1, "Section B", "different section"),
        Passage("p4", 2, "Section B", "different page"),
    ]
    _merge_short_passages(passages, min_words=50, max_words=250)
    assert [passage.passage_id for passage in passages] == ["p1", "p3", "p4"]
    assert passages[0].text == "short one short two"


def test_bbox_overlap_ratio() -> None:
    assert _bbox_overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _bbox_overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _bbox_overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5


def test_blank_page_is_reported_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    document, _page = _new_pdf(path)
    document.save(path)
    document.close()

    parsed = parse_pdf(path, extract_tables_enabled=False)
    assert parsed.page_count == 1
    assert parsed.passages == []
    assert len(parsed.unreadable_pages) == 1
    assert parsed.unreadable_pages[0].page == 1
    assert "scanned" in parsed.unreadable_pages[0].reason


def test_heading_and_body_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "simple.pdf"
    document, page = _new_pdf(path)
    page.insert_text((72, 72), "Climate Action", fontsize=18)
    body = (
        "The company has set a target to reduce absolute Scope 1 and Scope 2 "
        "greenhouse gas emissions by 42 per cent by 2030 against a FY2023 "
        "baseline. Progress is reported annually through the sustainability "
        "report and is reviewed by the board. The target covers all owned and "
        "operated sites within the reporting boundary. " * 2
    )
    page.insert_textbox(fitz.Rect(72, 110, 523, 420), body, fontsize=10)
    document.save(path)
    document.close()

    parsed = parse_pdf(path, extract_tables_enabled=False)
    assert parsed.page_count == 1
    assert parsed.unreadable_pages == []
    assert parsed.passages
    assert parsed.passages[0].page == 1
    assert parsed.passages[0].section_heading == "Climate Action"
    assert "greenhouse gas" in parsed.passages[0].text
    assert parsed.page_text(1)


def test_table_is_extracted_as_separate_passage(tmp_path: Path) -> None:
    path = tmp_path / "table.pdf"
    document, page = _new_pdf(path)

    # Draw a bordered 3x3 table and place text in each cell.
    x_positions = [72, 220, 360, 500]
    y_positions = [80, 120, 160, 200]
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]), width=1)
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y), width=1)

    cells = [
        ["Metric", "FY2023", "FY2024"],
        ["Scope 1 emissions", "1,000", "900"],
        ["Renewable electricity", "20%", "35%"],
    ]
    for row_index, row in enumerate(cells):
        for col_index, value in enumerate(row):
            page.insert_text(
                (x_positions[col_index] + 4, y_positions[row_index] + 24),
                value,
                fontsize=9,
            )

    document.save(path)
    document.close()

    parsed = parse_pdf(path, extract_tables_enabled=True)
    table_passages = [passage for passage in parsed.passages if passage.is_table]
    assert table_passages, "expected at least one table passage"
    assert "Scope 1 emissions" in table_passages[0].text
    assert table_passages[0].page == 1
    assert table_passages[0].is_table is True


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample BRSR PDF is not present")
def test_real_sample_first_pages_parse() -> None:
    parsed = parse_pdf(SAMPLE_PDF, extract_tables_enabled=False, max_pages=2)
    assert parsed.page_count == 2
    assert parsed.passages
    assert all(passage.page in (1, 2) for passage in parsed.passages)
    assert len({passage.passage_id for passage in parsed.passages}) == len(
        parsed.passages
    )
