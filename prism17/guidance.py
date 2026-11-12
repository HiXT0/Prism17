"""Deterministic disclosure guidance for the Phase 3 report.

Guidance is generated from the rubric dimensions only. It never claims that an
action did not happen; it describes what the *disclosure* does not yet contain.
No LLM call happens here.
"""

from __future__ import annotations

from typing import Sequence

from .coverage import GoalCoverage
from .models import VerifiedFinding
from .rubric import DIMENSION_KEYS

DIMENSION_SUGGESTIONS = {
    "specificity": (
        "Describe the specific action, policy, or commitment instead of a "
        "general statement."
    ),
    "measurable_indicator": (
        "Add a measurable indicator, such as a quantity, percentage, or "
        "intensity, with its unit and scope."
    ),
    "deadline": "State a target date or the reporting period for the commitment.",
    "baseline": "State a baseline year and baseline value for comparison.",
    "reported_progress": (
        "Report progress against the commitment, including the current result."
    ),
}

NEGATED_SUGGESTION = (
    "If the action is not applicable, state why. Otherwise describe the "
    "specific action and how it is measured."
)

NO_DISCLOSURE_GUIDANCE = (
    "No scored disclosure was identified for this goal among the reviewed "
    "candidates. This can reflect retrieval or judging limits rather than an "
    "absence of disclosure, so review the source report before drawing a "
    "conclusion. If the topic is material, a disclosure with a specific action, "
    "a measurable indicator, a target date, a baseline, and reported progress "
    "would support a higher rubric level."
)

COVERAGE_CAVEAT = (
    "Coverage counts are limited to the goals, targets, and candidates that "
    "were retrieved and judged in this run. A target without a scored finding "
    "is reported as not identified, not as evidence that the company does "
    "nothing."
)


def missing_dimensions(finding: VerifiedFinding) -> list[str]:
    """Return rubric dimensions the judge marked absent, in rubric order."""
    dimensions = finding.judged.dimensions or {}
    return [
        key for key in DIMENSION_KEYS if not bool(dimensions.get(key, False))
    ]


def guidance_for_finding(finding: VerifiedFinding) -> list[str]:
    """Suggestions for one scored finding, derived from its rubric gaps."""
    if not finding.is_scored:
        return []

    suggestions: list[str] = []
    if bool(finding.judged.extracted.get("negated", False)):
        suggestions.append(NEGATED_SUGGESTION)
    for key in missing_dimensions(finding):
        suggestions.append(DIMENSION_SUGGESTIONS[key])
    return _dedupe(suggestions)


def guidance_for_goal(goal: GoalCoverage) -> list[str]:
    """Goal-level guidance based on the best level reached in that goal."""
    if goal.finding_count == 0 or goal.best_level is None:
        return [NO_DISCLOSURE_GUIDANCE]

    if goal.best_level >= 3:
        return [
            "At least one disclosure for this goal includes a measurable, "
            "time-bound element. Check that the remaining targets in this goal "
            "are covered or explained."
        ]
    if goal.best_level == 2:
        return [
            "Disclosures for this goal are specific but incomplete. Where "
            "material, add a target date, a baseline, and reported progress."
        ]
    return [
        "Disclosures for this goal are general. Where material, describe the "
        "specific action and add a measurable indicator, target date, baseline, "
        "and reported progress."
    ]


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
