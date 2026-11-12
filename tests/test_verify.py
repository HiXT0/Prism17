"""Tests for stage 1 quote verification."""

from __future__ import annotations

import pytest

from prism17.models import Candidate, JudgedFinding
from prism17.parse import PageText, ParsedDocument
from prism17.verify import (
    normalise_for_match,
    verify_finding,
    verify_quote,
)


def _document(page_text: str) -> ParsedDocument:
    return ParsedDocument(
        pdf_path="example.pdf",
        title="example",
        page_count=1,
        pages=[PageText(page=1, text=page_text, char_count=len(page_text))],
    )


def _finding(quote: str, *, page: int = 1, relevant: bool = True) -> JudgedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=page,
        target_id="13.2",
        target_goal=13,
        target_text="Integrate climate change measures into policy.",
        score=0.9,
        passage_text="Example passage.",
    )
    return JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=13,
        sdg_target="13.2",
        claim="Example claim.",
        supporting_quote=quote,
        dimensions={},
    )


def test_normalise_repairs_line_break_hyphenation() -> None:
    assert normalise_for_match("cli-\nmate change") == "climate change"


def test_normalise_collapses_whitespace_and_smart_quotes() -> None:
    assert normalise_for_match("  the company\u2019s   plan \n") == "the company's plan"


def test_exact_quote_is_verified_on_page() -> None:
    page = (
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e."
    )
    check = verify_quote(
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024",
        page_text=page,
        passage_text=page,
    )
    assert check.verified
    assert check.method == "page_exact"
    assert check.score == 100.0


def test_quote_with_line_break_hyphenation_is_verified() -> None:
    page = "The company expanded its renew-\nable energy programme in 2024."
    check = verify_quote(
        "The company expanded its renewable energy programme in 2024.",
        page_text=page,
        passage_text=page,
    )
    assert check.verified
    assert check.method == "page_exact"


def test_fuzzy_quote_is_verified_above_threshold() -> None:
    pytest.importorskip("rapidfuzz")
    page = (
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e, achieving our 2030 target ahead of schedule."
    )
    check = verify_quote(
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e, achieving the 2030 target ahead of schedule.",
        page_text=page,
        passage_text=page,
    )
    assert check.verified
    assert check.method == "page_fuzzy"


def test_unrelated_quote_is_not_verified() -> None:
    check = verify_quote(
        "The company installed solar panels at three factories.",
        page_text="We reduced emissions by 42% against a baseline.",
        passage_text="We reduced emissions by 42% against a baseline.",
    )
    assert not check.verified
    assert check.score < 95


def test_table_passage_is_used_as_fallback() -> None:
    page = "Metric 2024 Emissions 120"
    passage = "Metric | FY2024 Emissions | 120"
    check = verify_quote(
        "Metric | FY2024 Emissions | 120",
        page_text=page,
        passage_text=passage,
        threshold=100.0,
    )
    assert check.verified
    assert check.method == "passage_exact"


def test_empty_or_short_quote_is_not_verified() -> None:
    assert not verify_quote("", page_text="text", passage_text="text").verified
    assert not verify_quote(
        "three words only", page_text="three words only", passage_text="three words only"
    ).verified


def test_verify_finding_marks_unverified_quote_for_review() -> None:
    document = _document("A completely different sentence about water use.")
    finding = _finding("We reduced emissions by 42% against a baseline.")
    verified = verify_finding(finding, document)
    assert not verified.quote_verified
    assert verified.review_reason is not None
    assert verified.source_page == 1


def test_verify_finding_passes_through_irrelevant_judgement() -> None:
    document = _document("Some page text.")
    verified = verify_finding(_finding("Some page text.", relevant=False), document)
    assert not verified.quote_verified
    assert verified.review_reason == "judge marked the passage as not relevant"


def test_verify_quote_handles_missing_source_text() -> None:
    check = verify_quote(
        "four words in this quote",
        page_text=None,
        passage_text=None,
    )
    assert not check.verified
    assert "no source text" in (check.reason or "")
