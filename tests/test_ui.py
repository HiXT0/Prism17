"""Tests for the Streamlit UI's pure helpers.

The Streamlit rendering itself is checked with a headless boot smoke test; these
tests cover the data transformations that must not drift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.report import build_report_data
from prism17.pipeline import PipelineResult
from prism17.parse import PageText, ParsedDocument, Passage
from prism17.review import finding_key, make_decision
from prism17.ui import (
    _decisions_from_frame,
    _default_review_path,
    _level_text,
    _review_dataframe,
    _short,
)
from prism17.config import load_config

QUOTE = (
    "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 against a "
    "FY2020 baseline of 100,000 tCO2e."
)


def _finding(*, page: int = 1, passage_id: str = "p0001-001") -> VerifiedFinding:
    candidate = Candidate(
        passage_id=passage_id,
        page=page,
        target_id="12.5",
        target_goal=12,
        target_text="Target text",
        score=0.9,
        passage_text=QUOTE,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=True,
        sdg_goal=12,
        sdg_target="12.5",
        claim="Example claim.",
        supporting_quote=QUOTE,
        dimensions={
            "specificity": True,
            "measurable_indicator": True,
            "deadline": False,
            "baseline": False,
            "reported_progress": False,
        },
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=True,
        quote_method="page_exact",
        quote_score=100.0,
        source_page=page,
        level=2,
        level_rationale="Level 2: specific with a measurable element.",
    )


def _data(findings: list[VerifiedFinding]):
    config = load_config()
    config["report"]["focus_sdgs"] = [12]
    document = ParsedDocument(
        pdf_path="reports/example.pdf",
        title="Example",
        page_count=1,
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
    result = PipelineResult(
        document=document,
        candidates=[finding.judged.candidate for finding in findings],
        judged=[finding.judged for finding in findings],
        findings=findings,
        config=config,
        elapsed_seconds=1.0,
    )
    return build_report_data(result)


def test_default_review_path_sits_next_to_audit() -> None:
    path = _default_review_path("outputs/sjs.phase2.json")

    assert path == Path("outputs/sjs.phase2.reviews.json")
    assert _default_review_path(None) is None


def test_level_text_handles_missing_and_present() -> None:
    assert _level_text(None) == "—"
    assert _level_text("") == "—"
    assert _level_text(3) == "Level 3"


def test_short_squashes_and_truncates() -> None:
    assert _short("a  b\n c") == "a b c"
    shortened = _short("word " * 100, limit=20)
    assert len(shortened) <= 20
    assert shortened.endswith("…")


def test_review_dataframe_has_one_row_per_scored_finding() -> None:
    data = _data([_finding(), _finding(page=2, passage_id="p0002-001")])

    frame = _review_dataframe(data, {})

    assert len(frame) == 2
    assert set(frame["review_status"]) == {"pending"}
    assert frame["review_level"].isna().all()


def test_review_dataframe_merges_existing_decisions() -> None:
    finding = _finding()
    key = finding_key(finding)
    data = _data([finding])
    decisions = {
        key: make_decision(
            key, status="overridden", level=4, reviewer="Reviewer A"
        )
    }

    frame = _review_dataframe(data, decisions)

    assert frame.iloc[0]["review_status"] == "overridden"
    assert frame.iloc[0]["review_level"] == 4
    assert frame.iloc[0]["reviewer"] == "Reviewer A"


def test_decisions_from_frame_roundtrips_edits() -> None:
    finding = _finding()
    key = finding_key(finding)
    frame = pd.DataFrame(
        [
            {
                "finding_key": key,
                "target": "12.5",
                "page": 1,
                "review_status": "overridden",
                "review_level": 3,
                "note": "Baseline appears in the table.",
                "reviewer": "Reviewer A",
            }
        ]
    )

    decisions, errors = _decisions_from_frame(frame)

    assert errors == []
    assert decisions[key].status == "overridden"
    assert decisions[key].level == 3
    assert decisions[key].note == "Baseline appears in the table."


def test_override_without_level_is_an_error() -> None:
    finding = _finding()
    key = finding_key(finding)
    frame = pd.DataFrame(
        [
            {
                "finding_key": key,
                "target": "12.5",
                "page": 1,
                "review_status": "overridden",
                "review_level": None,
                "note": "",
                "reviewer": "",
            }
        ]
    )

    decisions, errors = _decisions_from_frame(frame)

    assert key not in decisions
    assert len(errors) == 1
    assert "choose a reviewer level" in errors[0]


def test_confirmed_status_drops_a_stray_level() -> None:
    finding = _finding()
    key = finding_key(finding)
    frame = pd.DataFrame(
        [
            {
                "finding_key": key,
                "target": "12.5",
                "page": 1,
                "review_status": "confirmed",
                "review_level": 4,
                "note": "",
                "reviewer": "",
            }
        ]
    )

    decisions, errors = _decisions_from_frame(frame)

    assert errors == []
    assert decisions[key].status == "confirmed"
    assert decisions[key].level is None
