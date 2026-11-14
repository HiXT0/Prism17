"""Tests for Phase 3 Markdown and HTML report rendering."""

from __future__ import annotations

import json

from prism17.config import load_config
from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.parse import PageText, ParsedDocument, Passage
from prism17.pipeline import PipelineResult
from prism17.report import (
    build_report_data,
    render_html,
    render_markdown,
    report_data_from_json,
    write_html_report,
    write_markdown_report,
)
from prism17.review import finding_key, make_decision, save_reviews

QUOTE = (
    "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
    "FY2020 baseline of 100,000 tCO2e."
)


def _config() -> dict:
    config = load_config()
    config["report"]["focus_sdgs"] = [12]
    return config


def _document() -> ParsedDocument:
    return ParsedDocument(
        pdf_path="reports/example.pdf",
        title="Example Report",
        page_count=2,
        passages=[
            Passage(
                passage_id="p0001-001",
                page=1,
                section_heading=None,
                text=QUOTE,
            )
        ],
        pages=[PageText(page=1, text=QUOTE, char_count=len(QUOTE))],
    )


def _finding(
    *,
    relevant: bool = True,
    quote_verified: bool = True,
    level: int | None = 2,
    review_reason: str | None = None,
    page: int = 1,
    quote: str = QUOTE,
    claim: str = "Example claim.",
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=page,
        target_id="12.5",
        target_goal=12,
        target_text="Reduce waste generation through prevention and recycling.",
        score=0.9,
        passage_text=QUOTE,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=12,
        sdg_target="12.5",
        claim=claim,
        supporting_quote=quote,
        dimensions={
            "specificity": True,
            "measurable_indicator": True,
            "deadline": False,
            "baseline": False,
            "reported_progress": False,
        },
        extracted={"negated": False},
        scored=review_reason is None,
    )
    scored = quote_verified and relevant and review_reason is None
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=page,
        level=level if scored else None,
        level_rationale="Level 2: specific disclosure with a measurable element.",
        disclosure_gaps=["No target date or deadline is disclosed."],
        review_reason=review_reason,
    )


def _result(findings: list[VerifiedFinding]) -> PipelineResult:
    return PipelineResult(
        document=_document(),
        candidates=[finding.judged.candidate for finding in findings],
        judged=[finding.judged for finding in findings],
        findings=findings,
        config=_config(),
        elapsed_seconds=1.0,
    )


def test_markdown_report_has_page_quote_and_level() -> None:
    data = build_report_data(_result([_finding()]))

    markdown = render_markdown(data)

    assert "SDG Alignment Report — Example Report" in markdown
    assert "Page 1" in markdown
    assert "Level 2" in markdown
    assert "42% in FY2024" in markdown
    assert "1 of 11 focus targets" in markdown
    assert "disclosure quality" in markdown
    assert "## Method and limitations" in markdown


def test_html_report_escapes_untrusted_text() -> None:
    dangerous = '<script>alert("x")</script> with enough words to verify.'
    data = build_report_data(
        _result([_finding(claim=dangerous, quote=dangerous)])
    )

    html_text = render_html(data)

    assert "<script>alert" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<blockquote>" in html_text
    assert "Page 1" in html_text


def test_json_roundtrip_preserves_scored_finding() -> None:
    result = _result([_finding()])
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))

    data = report_data_from_json(payload, _config())

    assert len(data.findings) == 1
    assert len(data.scored) == 1
    assert data.scored[0].source_page == 1
    assert data.scored[0].judged.supporting_quote == QUOTE
    assert data.coverage.targets_with_findings == 1


def test_json_without_findings_still_reports_coverage() -> None:
    data = report_data_from_json(
        {
            "document": {"title": "Empty", "page_count": 1},
            "settings": {"focus_sdgs": [12]},
            "stats": {},
            "findings": [],
        },
        _config(),
    )

    assert data.findings == []
    assert data.coverage.targets_total == 11
    assert "No scored findings" in render_markdown(data)


def test_needs_review_is_rendered_separately() -> None:
    scored = _finding()
    review = _finding(
        page=2,
        quote_verified=False,
        level=None,
        review_reason=(
            "unscored (needs human review): quote could not be verified"
        ),
    )
    data = build_report_data(_result([scored, review]))

    markdown = render_markdown(data)

    assert data.needs_review == [review]
    assert "## Needs human review" in markdown
    assert "Page 2, target 12.5" in markdown
    assert "quote could not be verified" in markdown


def test_no_scored_findings_renders_goal_guidance() -> None:
    data = build_report_data(
        _result(
            [
                _finding(
                    relevant=False,
                    quote_verified=False,
                    level=None,
                )
            ]
        )
    )

    markdown = render_markdown(data)

    assert "No scored findings" in markdown
    assert "## Goals without scored findings" in markdown
    assert "retrieval or judging limits" in markdown


def test_write_reports_creates_both_files(tmp_path) -> None:
    data = build_report_data(_result([_finding()]))

    markdown_path = write_markdown_report(
        data, tmp_path / "report.md", _config()
    )
    html_path = write_html_report(data, tmp_path / "report.html", _config())

    assert "SDG Alignment Report" in markdown_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_path.read_text(encoding="utf-8")


def test_markdown_report_includes_review_decision() -> None:
    finding = _finding()
    data = build_report_data(_result([finding]))
    key = finding_key(finding)
    data.reviews = {
        key: make_decision(
            key,
            status="overridden",
            level=3,
            note="The prior-year table shows a baseline.",
            reviewer="Reviewer A",
        )
    }

    markdown = render_markdown(data)

    assert "Overridden to Level 3 by Reviewer A" in markdown
    assert "## Human review" in markdown
    assert "| 12.5 | 1 | Level 2 | overridden | Level 3 |" in markdown
    assert "The prior-year table shows a baseline." in markdown


def test_html_report_includes_review_decision() -> None:
    finding = _finding()
    data = build_report_data(_result([finding]))
    key = finding_key(finding)
    data.reviews = {
        key: make_decision(key, status="confirmed", reviewer="Reviewer B")
    }

    html_text = render_html(data)

    assert "<h2>Human review</h2>" in html_text
    assert "Confirmed by Reviewer B" in html_text
    assert "Reviewer B" in html_text


def test_report_without_reviews_has_no_review_table() -> None:
    data = build_report_data(_result([_finding()]))

    markdown = render_markdown(data)
    html_text = render_html(data)

    assert "## Human review" not in markdown
    assert "<h2>Human review</h2>" not in html_text
    assert "Not reviewed (1 pending)" in markdown


def test_pending_decisions_do_not_render_a_review_table() -> None:
    finding = _finding()
    data = build_report_data(_result([finding]))
    key = finding_key(finding)
    data.reviews = {key: make_decision(key, status="pending")}

    markdown = render_markdown(data)

    assert "## Human review" not in markdown
    assert "Not reviewed (1 pending)" in markdown


def test_reports_include_the_evidence_coverage_matrix() -> None:
    data = build_report_data(_result([_finding()]))

    markdown = render_markdown(data)
    html_text = render_html(data)

    assert "## Evidence coverage matrix" in markdown
    assert (
        "| Goal 12 | 1 | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) "
        "| 0/1 (0%) |"
    ) in markdown
    assert "| All focus goals | 1 | 1/1 (100%)" in markdown
    assert "a disclosure observation, not evidence" in markdown

    assert "<h2>Evidence coverage matrix</h2>" in html_text
    assert "<th>Specificity</th>" in html_text
    assert "<th>Measurable indicator</th>" in html_text
    assert "<th>Reported progress</th>" in html_text
    assert "0/1 (0%)" in html_text
    assert "All focus goals" in html_text


def test_evidence_matrix_without_focus_goals_renders_a_message() -> None:
    config = _config()
    config["report"]["focus_sdgs"] = []
    data = report_data_from_json(
        {
            "document": {"title": "Empty", "page_count": 1},
            "settings": {"focus_sdgs": []},
            "stats": {},
            "findings": [],
        },
        config,
    )

    assert "No scored findings were available for dimension accounting." in (
        render_markdown(data)
    )
    assert "No scored findings were available for dimension accounting." in (
        render_html(data)
    )


def test_report_cli_overlays_review_decisions(tmp_path) -> None:
    from prism17.report import main as report_main

    finding = _finding()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(_result([finding]).to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    key = finding_key(finding)
    reviews_path = save_reviews(
        tmp_path / "reviews.json",
        {
            key: make_decision(
                key,
                status="overridden",
                level=4,
                note="Reviewed in the UI.",
                reviewer="Reviewer A",
            )
        },
    )
    markdown_path = tmp_path / "report.md"

    code = report_main(
        [
            str(audit_path),
            "--reviews",
            str(reviews_path),
            "--out",
            str(markdown_path),
        ]
    )

    assert code == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Overridden to Level 4 by Reviewer A" in markdown
    assert "Reviewed in the UI." in markdown
