"""Tests for deterministic Phase 3 disclosure guidance."""

from __future__ import annotations

from prism17.coverage import GoalCoverage
from prism17.guidance import (
    NEGATED_SUGGESTION,
    NO_DISCLOSURE_GUIDANCE,
    guidance_for_finding,
    guidance_for_goal,
    missing_dimensions,
)
from prism17.models import Candidate, JudgedFinding, VerifiedFinding

ALL_DIMENSIONS = {
    "specificity": True,
    "measurable_indicator": True,
    "deadline": True,
    "baseline": True,
    "reported_progress": True,
}


def _finding(
    dimensions: dict[str, bool] | None = None,
    *,
    level: int | None = 2,
    negated: bool = False,
    scored: bool = True,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=1,
        target_id="13.2",
        target_goal=13,
        target_text="Target text",
        score=0.9,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=scored,
        sdg_goal=13,
        sdg_target="13.2",
        claim="Example claim.",
        supporting_quote="Example quote with enough words to verify.",
        dimensions=dict(dimensions if dimensions is not None else ALL_DIMENSIONS),
        extracted={"negated": negated},
        scored=scored,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=scored,
        quote_method="page_exact" if scored else None,
        quote_score=100.0 if scored else 0.0,
        source_page=1,
        level=level if scored else None,
    )


def _goal(**overrides: object) -> GoalCoverage:
    values: dict[str, object] = {
        "goal": 13,
        "short_title": "Climate Action",
        "goal_title": "Take urgent action to combat climate change.",
        "targets_total": 5,
    }
    values.update(overrides)
    return GoalCoverage(**values)  # type: ignore[arg-type]


def test_missing_dimensions_returns_rubric_order() -> None:
    finding = _finding(
        {
            "specificity": True,
            "measurable_indicator": False,
            "deadline": False,
            "baseline": True,
            "reported_progress": True,
        }
    )

    assert missing_dimensions(finding) == ["measurable_indicator", "deadline"]


def test_guidance_for_finding_maps_each_gap() -> None:
    finding = _finding(
        {
            "specificity": True,
            "measurable_indicator": True,
            "deadline": False,
            "baseline": False,
            "reported_progress": False,
        }
    )

    suggestions = guidance_for_finding(finding)

    assert len(suggestions) == 3
    assert any("target date" in item for item in suggestions)
    assert any("baseline" in item for item in suggestions)
    assert any("progress" in item for item in suggestions)


def test_guidance_for_complete_disclosure_is_empty() -> None:
    assert guidance_for_finding(_finding(level=4)) == []


def test_guidance_for_unscored_finding_is_empty() -> None:
    assert guidance_for_finding(_finding(scored=False)) == []


def test_negated_disclosure_gets_specific_suggestion() -> None:
    suggestions = guidance_for_finding(_finding(negated=True, level=1))

    assert NEGATED_SUGGESTION in suggestions


def test_goal_without_findings_mentions_retrieval_limits() -> None:
    guidance = guidance_for_goal(_goal())

    assert guidance == [NO_DISCLOSURE_GUIDANCE]
    assert "retrieval or judging limits" in guidance[0]


def test_goal_level_two_guidance_asks_for_supporting_detail() -> None:
    guidance = guidance_for_goal(
        _goal(
            finding_count=1,
            best_level=2,
            status="partial disclosure",
        )
    )

    assert "target date" in guidance[0]
    assert "baseline" in guidance[0]


def test_goal_level_four_guidance_is_neutral() -> None:
    guidance = guidance_for_goal(
        _goal(
            finding_count=1,
            best_level=4,
            status="quantified disclosure",
        )
    )

    assert "remaining targets" in guidance[0]
