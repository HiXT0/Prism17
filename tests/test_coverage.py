"""Tests for Phase 3 coverage accounting."""

from __future__ import annotations

from prism17.config import load_config
from prism17.coverage import build_coverage, target_sort_key
from prism17.models import Candidate, JudgedFinding, VerifiedFinding

DIMENSIONS = {
    "specificity": True,
    "measurable_indicator": True,
    "deadline": True,
    "baseline": True,
    "reported_progress": True,
}


def _finding(
    target_id: str,
    level: int | None,
    *,
    page: int = 10,
    goal: int = 12,
    quote_verified: bool = True,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id=f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=goal,
        target_text="Target text",
        score=0.8,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=True,
        sdg_goal=goal,
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote="Example quote with enough words to verify.",
        dimensions=dict(DIMENSIONS),
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=page,
        level=level if quote_verified else None,
    )


def test_build_coverage_counts_goals_and_targets() -> None:
    findings = [
        _finding("12.5", 2, page=49),
        _finding("12.5", 1, page=50),
        _finding("12.6", 1, page=25),
    ]

    summary = build_coverage(findings, load_config(), focus_goals=[12])

    assert len(summary.goals) == 1
    goal = summary.goals[0]
    assert goal.goal == 12
    assert goal.targets_total == 11
    assert goal.covered_target_count == 2
    assert goal.finding_count == 3
    assert goal.best_level == 2
    assert goal.level_counts == {1: 2, 2: 1, 3: 0, 4: 0}
    assert goal.status == "partial disclosure"
    assert summary.targets_with_findings == 2

    target = next(item for item in goal.targets if item.target_id == "12.5")
    assert target.finding_count == 2
    assert target.pages == [49, 50]
    assert target.best_level == 2
    assert target.status == "partial disclosure"


def test_build_coverage_ignores_unscored_findings() -> None:
    summary = build_coverage(
        [_finding("12.5", 2, quote_verified=False)],
        load_config(),
        focus_goals=[12],
    )

    assert summary.targets_with_findings == 0
    assert summary.goals[0].status == "no scored disclosure identified"
    assert summary.goals[0].finding_count == 0


def test_build_coverage_status_for_each_level() -> None:
    expected = {
        1: "general disclosure only",
        2: "partial disclosure",
        3: "quantified disclosure",
        4: "quantified disclosure",
    }
    for level, status in expected.items():
        summary = build_coverage(
            [_finding("12.5", level)], load_config(), focus_goals=[12]
        )
        assert summary.goals[0].status == status


def test_build_coverage_uses_configured_focus_goals() -> None:
    summary = build_coverage([_finding("8.8", 2, goal=8)], load_config())

    assert {goal.goal for goal in summary.goals} == {8, 12, 13, 17}
    assert summary.goals_with_findings[0].goal == 8


def test_build_coverage_falls_back_to_finding_goals() -> None:
    config = load_config()
    config["report"]["focus_sdgs"] = []

    summary = build_coverage(
        [_finding("8.8", 2, goal=8)], config, focus_goals=None
    )

    assert [goal.goal for goal in summary.goals] == [8]


def test_target_sort_key_orders_numerically() -> None:
    assert target_sort_key("12.2") < target_sort_key("12.10")
    assert target_sort_key("12.10") < target_sort_key("12.a")
