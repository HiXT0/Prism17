"""Tests for deterministic pipeline helpers and wiring."""

from __future__ import annotations

import pytest

from prism17.config import load_config
from prism17.models import Candidate, JudgedFinding
from prism17.parse import PageText, ParsedDocument
from prism17.pipeline import (
    _parse_goals,
    _select_candidates,
    main,
    run_pipeline,
)


def _candidate(passage_id: str, score: float) -> Candidate:
    return Candidate(
        passage_id=passage_id,
        page=1,
        target_id="13.2",
        target_goal=13,
        target_text="Target",
        score=score,
        passage_text="Passage",
    )


def _document(page_text: str) -> ParsedDocument:
    return ParsedDocument(
        pdf_path="example.pdf",
        title="example",
        page_count=1,
        pages=[PageText(page=1, text=page_text, char_count=len(page_text))],
    )


def _judged(
    quote: str,
    *,
    relevant: bool = True,
    passage_text: str | None = None,
    scored: bool = True,
    review_reason: str | None = None,
) -> JudgedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=1,
        target_id="13.2",
        target_goal=13,
        target_text="Integrate climate change measures into policy.",
        score=0.9,
        passage_text=passage_text if passage_text is not None else quote,
    )
    return JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=13,
        sdg_target="13.2",
        claim="Example claim.",
        supporting_quote=quote,
        dimensions={
            "specificity": True,
            "measurable_indicator": True,
            "deadline": True,
            "baseline": True,
            "reported_progress": True,
        },
        extracted={"negated": False},
        scored=scored,
        review_reason=review_reason,
    )


def test_select_candidates_sorts_by_score() -> None:
    selected = _select_candidates(
        [_candidate("low", 0.4), _candidate("high", 0.9), _candidate("mid", 0.6)],
        max_candidates=10,
    )
    assert [item.passage_id for item in selected] == ["high", "mid", "low"]


def test_select_candidates_applies_limit() -> None:
    selected = _select_candidates(
        [_candidate("a", 0.4), _candidate("b", 0.9), _candidate("c", 0.6)],
        max_candidates=2,
    )
    assert [item.passage_id for item in selected] == ["b", "c"]


def test_select_candidates_none_means_no_limit() -> None:
    selected = _select_candidates([_candidate("a", 0.4)], max_candidates=None)
    assert len(selected) == 1


def test_parse_goals_accepts_spaces_and_commas() -> None:
    assert _parse_goals(["8", "12,13", "17"]) == [8, 12, 13, 17]


def test_parse_goals_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _parse_goals(["18"])


def test_parse_goals_none() -> None:
    assert _parse_goals(None) is None


def test_run_pipeline_scores_verified_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = (
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e."
    )
    document = _document(quote)
    candidate = _judged(quote).candidate

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [_judged(quote)]
    )

    result = run_pipeline("example.pdf", load_config(), max_candidates=10)

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.is_scored
    assert finding.level == 4
    assert finding.source_page == 1
    assert finding.quote_method == "page_exact"


def test_run_pipeline_never_scores_unverified_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_text = "A completely different sentence about water use."
    document = _document(page_text)
    quote = "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024."
    judged = _judged(quote, passage_text=page_text)
    candidate = judged.candidate

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [judged]
    )

    result = run_pipeline("example.pdf", load_config(), max_candidates=10)

    assert result.findings
    assert not result.scored
    assert result.needs_review
    assert result.findings[0].level is None


def test_run_pipeline_counts_not_relevant_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024."
    document = _document(quote)
    judged = _judged(quote, relevant=False)
    candidate = judged.candidate

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [judged]
    )

    result = run_pipeline("example.pdf", load_config(), max_candidates=10)

    assert not result.scored
    assert not result.needs_review
    assert len(result.not_relevant) == 1
    stats = result.stats()
    assert stats["not_relevant"] == 1
    assert stats["needs_review"] == 0
    assert stats["scored"] == 0


def test_run_pipeline_counts_schema_failure_as_needs_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024."
    document = _document(quote)
    judged = _judged(
        quote,
        relevant=False,
        scored=False,
        review_reason="judge output failed schema validation",
    )
    candidate = judged.candidate

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [judged]
    )

    result = run_pipeline("example.pdf", load_config(), max_candidates=10)

    assert not result.scored
    assert not result.not_relevant
    assert len(result.needs_review) == 1
    assert "schema" in (result.needs_review[0].review_reason or "")
    stats = result.stats()
    assert stats["not_relevant"] == 0
    assert stats["needs_review"] == 1


def test_run_pipeline_records_effective_semantic_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = (
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e."
    )
    document = _document(quote)
    candidate = _judged(quote).candidate
    calls: list[bool] = []

    def _fake_apply(findings, config, **kwargs):
        calls.append(True)
        for finding in findings:
            finding.semantic_support = "supported"
            finding.semantic_score = 0.9
        return list(findings)

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [_judged(quote)]
    )
    monkeypatch.setattr(
        "prism17.pipeline.apply_semantic_support", _fake_apply
    )

    config = load_config()
    config["verification"]["semantic_enabled"] = False
    result = run_pipeline(
        "example.pdf", config, max_candidates=10, semantic_check=True
    )

    assert calls == [True]
    assert result.to_dict()["settings"]["semantic_enabled"] is True
    assert result.findings[0].semantic_support == "supported"


def test_main_writes_audit_json_and_reports(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = (
        "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
        "FY2020 baseline of 100,000 tCO2e."
    )
    document = _document(quote)
    judged = _judged(quote)
    candidate = judged.candidate

    monkeypatch.setattr("prism17.pipeline.parse_pdf", lambda *a, **k: document)
    monkeypatch.setattr(
        "prism17.pipeline.retrieve_candidates", lambda *a, **k: [candidate]
    )
    monkeypatch.setattr(
        "prism17.pipeline.judge_candidates", lambda *a, **k: [judged]
    )

    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"
    code = main(
        [
            str(tmp_path / "example.pdf"),
            "--out",
            str(audit_path),
            "--report",
            str(report_path),
            "--html-report",
            str(html_path),
        ]
    )

    assert code == 0
    assert audit_path.exists()
    assert "SDG Alignment Report" in report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_path.read_text(encoding="utf-8")
