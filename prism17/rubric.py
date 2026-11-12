"""Deterministic rubric for disclosure quality.

Levels describe the strength of the *disclosure*, not the company's real-world
performance. The rules use only the five boolean dimensions returned by the
judge, plus the explicit ``negated`` flag. No LLM call happens here.

Precedence (first matching rule wins):

1. The passage says the action is not applicable, not done, or not measured
   -> Level 1.
2. No specific action/commitment -> Level 1.
3. Specific + measurable + deadline + baseline + reported progress -> Level 4.
4. Specific + measurable + deadline -> Level 3.
5. Specific + at least one of measurable/deadline/baseline/progress -> Level 2.
6. Otherwise -> Level 1.

Every branch is covered by tests. Change the rules only together with the
README rubric documentation.
"""

from __future__ import annotations

from typing import Any, Sequence

from .config import get
from .models import VerifiedFinding

DIMENSION_KEYS = (
    "specificity",
    "measurable_indicator",
    "deadline",
    "baseline",
    "reported_progress",
)

GAP_TEXT = {
    "specificity": "The disclosure is general rather than specific.",
    "measurable_indicator": "No measurable indicator is disclosed.",
    "deadline": "No target date or deadline is disclosed.",
    "baseline": "No baseline or prior-year comparison is disclosed.",
    "reported_progress": "No reported progress or result is disclosed.",
}

NEGATED_GAP = (
    "The report states the action is not applicable, not done, or not measured."
)


def _dimensions(finding: VerifiedFinding) -> dict[str, bool]:
    raw = finding.judged.dimensions or {}
    return {key: bool(raw.get(key, False)) for key in DIMENSION_KEYS}


def _disclosure_gaps(dimensions: dict[str, bool]) -> list[str]:
    return [GAP_TEXT[key] for key in DIMENSION_KEYS if not dimensions[key]]


def assign_level(
    dimensions: dict[str, bool],
    *,
    negated: bool = False,
) -> tuple[int, str, list[str]]:
    """Return ``(level, rationale, disclosure_gaps)`` for one finding."""
    dims = {key: bool(dimensions.get(key, False)) for key in DIMENSION_KEYS}
    gaps = _disclosure_gaps(dims)

    if negated:
        return (
            1,
            "Level 1: the report states the action is not applicable, "
            "not done, or not measured.",
            [NEGATED_GAP, *gaps],
        )

    if not dims["specificity"]:
        return (
            1,
            "Level 1: relevant but general disclosure without a concrete "
            "action, commitment, or measurable element.",
            gaps,
        )

    if all(dims[key] for key in DIMENSION_KEYS):
        return (
            4,
            "Level 4: specific disclosure with a measurable indicator, "
            "deadline, baseline, and reported progress.",
            [],
        )

    if dims["measurable_indicator"] and dims["deadline"]:
        return (
            3,
            "Level 3: specific disclosure with a measurable indicator and "
            "deadline, but not all baseline and progress evidence.",
            gaps,
        )

    if any(
        dims[key]
        for key in (
            "measurable_indicator",
            "deadline",
            "baseline",
            "reported_progress",
        )
    ):
        return (
            2,
            "Level 2: specific disclosure with at least one of a measurable "
            "indicator, deadline, baseline, or reported progress.",
            gaps,
        )

    return (
        1,
        "Level 1: specific action but no measurable, time-bound, baseline, "
        "or progress information.",
        gaps,
    )


def grade_finding(finding: VerifiedFinding) -> VerifiedFinding:
    """Attach a rubric level to a verified finding.

    Unverified findings and findings already flagged for review keep
    ``level=None`` and can never be scored.
    """
    if (
        not finding.quote_verified
        or finding.review_reason is not None
        or not finding.judged.relevant
        or not finding.judged.scored
    ):
        finding.level = None
        finding.level_rationale = ""
        finding.disclosure_gaps = []
        return finding

    dimensions = _dimensions(finding)
    negated = bool(finding.judged.extracted.get("negated", False))
    level, rationale, gaps = assign_level(dimensions, negated=negated)
    finding.level = level
    finding.level_rationale = rationale
    finding.disclosure_gaps = gaps
    return finding


def grade_findings(
    findings: Sequence[VerifiedFinding],
    config: dict[str, Any] | None = None,
) -> list[VerifiedFinding]:
    """Grade a sequence of verified findings.

    ``config`` is accepted for a consistent pipeline API. The rubric itself has
    no tunable thresholds in this phase.
    """
    _ = get(config or {}, "rubric", default={})
    return [grade_finding(finding) for finding in findings]
