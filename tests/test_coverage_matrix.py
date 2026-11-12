"""Tests for the five-dimension evidence coverage matrix."""

from __future__ import annotations

from prism17.coverage import (
    ABSENT,
    NOT_ASSESSABLE,
    PRESENT,
    build_coverage_matrix,
)
from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.rubric import DIMENSION_KEYS

FULL = {
    "specificity": True,
    "measurable_indicator": True,
    "deadline": True,
    "baseline": True,
    "reported_progress": True,
}


def _finding(
    target_id: str = "12.5",
    *,
    goal: int = 12,
    page: int = 49,
    level: int | None = 3,
    dimensions: dict | None = None,
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
    review_reason: str | None = None,
    passage_id: str | None = None,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id=passage_id or f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=goal,
        target_text="Target text",
        score=0.8,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=goal,
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote="Example quote with enough words to verify.",
        dimensions=dict(FULL if dimensions is None else dimensions),
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=page,
        semantic_support=semantic_support,
        level=level if quote_verified else None,
        review_reason=review_reason,
    )


def test_scored_claim_marks_present_and_absent_dimensions() -> None:
    dimensions = dict(FULL)
    dimensions["specificity"] = False
    dimensions["deadline"] = False
    matrix = build_coverage_matrix([_finding(dimensions=dimensions)])

    assert len(matrix.claims) == 1
    claim = matrix.claims[0]
    assert claim.scored is True
    assert claim.dimensions["specificity"] == ABSENT
    assert claim.dimensions["deadline"] == ABSENT
    assert claim.dimensions["baseline"] == PRESENT
    assert claim.flags == ["disclosure_gap", "low_specificity", "no_deadline"]


def test_scored_claim_with_no_gaps_has_no_flags() -> None:
    matrix = build_coverage_matrix([_finding()])
    assert matrix.claims[0].flags == []


def test_unverified_claim_is_not_assessable() -> None:
    matrix = build_coverage_matrix([_finding(quote_verified=False)])
    claim = matrix.claims[0]
    assert claim.scored is False
    assert claim.quote_verified is False
    assert set(claim.dimensions.values()) == {NOT_ASSESSABLE}
    assert claim.flags == ["not_assessable"]


def test_semantic_uncertainty_is_not_assessable() -> None:
    matrix = build_coverage_matrix([_finding(semantic_support="uncertain")])
    claim = matrix.claims[0]
    assert claim.scored is False
    assert claim.semantic_support == "uncertain"
    assert claim.flags == ["not_assessable"]


def test_clean_not_relevant_candidate_is_excluded() -> None:
    matrix = build_coverage_matrix([_finding(relevant=False)])
    assert matrix.claims == []


def test_unscored_but_reviewable_candidate_is_kept_and_flagged() -> None:
    matrix = build_coverage_matrix(
        [_finding(level=None, review_reason="unscored (needs human review): test")]
    )
    assert len(matrix.claims) == 1
    assert matrix.claims[0].scored is False
    assert matrix.claims[0].flags == ["not_assessable"]


def test_dimension_and_flag_counts_aggregate_claims() -> None:
    dimensions = dict(FULL)
    dimensions["baseline"] = False
    matrix = build_coverage_matrix(
        [
            _finding(dimensions=dimensions, page=49),
            _finding(quote_verified=False, page=50),
        ]
    )

    counts = matrix.dimension_counts()
    assert set(counts) == set(DIMENSION_KEYS)
    assert counts["baseline"][ABSENT] == 1
    assert counts["baseline"][NOT_ASSESSABLE] == 1
    assert counts["deadline"][PRESENT] == 1

    flags = matrix.flag_counts()
    assert flags["disclosure_gap"] == 1
    assert flags["not_assessable"] == 1


def test_matrix_is_sorted_by_goal_target_and_page() -> None:
    matrix = build_coverage_matrix(
        [
            _finding("12.10", page=2, passage_id="p0002-001"),
            _finding("12.2", page=9, passage_id="p0009-001"),
            _finding("12.2", page=4, passage_id="p0004-001"),
        ]
    )
    assert [(claim.target_id, claim.page) for claim in matrix.claims] == [
        ("12.2", 4),
        ("12.2", 9),
        ("12.10", 2),
    ]


def test_matrix_to_dict_includes_labels_and_flag_text() -> None:
    dimensions = dict(FULL)
    dimensions["deadline"] = False
    data = build_coverage_matrix([_finding(dimensions=dimensions)]).to_dict()

    assert data["dimension_labels"]["deadline"] == "Deadline"
    assert data["flag_text"]["no_deadline"] == "disclosure gap: no deadline"
    assert data["claims"][0]["flag_text"] == [
        "disclosure gap",
        "disclosure gap: no deadline",
    ]
