"""Tests for the deterministic disclosure rubric."""

from __future__ import annotations

import pytest

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.rubric import assign_level, grade_finding


def _finding(
    dimensions: dict[str, bool],
    *,
    negated: bool = False,
    quote_verified: bool = True,
    review_reason: str | None = None,
    relevant: bool = True,
    scored: bool = True,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=1,
        target_id="13.2",
        target_goal=13,
        target_text="Integrate climate change measures into policy.",
        score=0.9,
        passage_text="Example passage.",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=13,
        sdg_target="13.2",
        claim="Example claim.",
        supporting_quote="Example passage with enough words to verify.",
        dimensions=dimensions,
        extracted={"negated": negated},
        scored=scored,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=1,
        review_reason=review_reason,
    )


def _dims(**overrides: bool) -> dict[str, bool]:
    values = {
        "specificity": False,
        "measurable_indicator": False,
        "deadline": False,
        "baseline": False,
        "reported_progress": False,
    }
    values.update(overrides)
    return values


def test_level_4_requires_all_five_dimensions() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            )
        )
    )
    assert finding.level == 4
    assert finding.disclosure_gaps == []
    assert finding.is_scored


def test_level_3_requires_specific_measurable_deadline() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
            )
        )
    )
    assert finding.level == 3
    assert finding.disclosure_gaps


@pytest.mark.parametrize(
    "extra",
    ["measurable_indicator", "deadline", "baseline", "reported_progress"],
)
def test_level_2_accepts_one_supporting_dimension(extra: str) -> None:
    finding = grade_finding(_finding(_dims(specificity=True, **{extra: True})))
    assert finding.level == 2


def test_level_1_when_disclosure_is_not_specific() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=False,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            )
        )
    )
    assert finding.level == 1
    assert finding.disclosure_gaps


def test_level_1_when_specific_but_no_other_evidence() -> None:
    finding = grade_finding(_finding(_dims(specificity=True)))
    assert finding.level == 1


def test_negated_disclosure_is_level_1_even_if_dimensions_are_true() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            ),
            negated=True,
        )
    )
    assert finding.level == 1
    assert any("not applicable" in gap for gap in finding.disclosure_gaps)


def test_unverified_finding_is_never_scored() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            ),
            quote_verified=False,
        )
    )
    assert finding.level is None
    assert not finding.is_scored


def test_review_reason_blocks_scoring() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            ),
            review_reason="needs review",
        )
    )
    assert finding.level is None
    assert not finding.is_scored


def test_irrelevant_finding_is_never_scored() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            ),
            relevant=False,
        )
    )
    assert finding.level is None
    assert not finding.is_scored


def test_unscored_judgement_is_never_scored() -> None:
    finding = grade_finding(
        _finding(
            _dims(
                specificity=True,
                measurable_indicator=True,
                deadline=True,
                baseline=True,
                reported_progress=True,
            ),
            scored=False,
        )
    )
    assert finding.level is None
    assert not finding.is_scored


def test_assign_level_is_deterministic() -> None:
    dims = _dims(specificity=True, measurable_indicator=True, deadline=True)
    first = assign_level(dims)
    second = assign_level(dims)
    assert first == second
