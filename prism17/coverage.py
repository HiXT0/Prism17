"""SDG coverage accounting for the Phase 3 report.

Coverage is a disclosure measure: it counts targets that have at least one
scored finding. A target with no scored finding is reported as "no scored
disclosure identified", never as proof that the company does nothing. Retrieval
and judging limits are restated in the report's method section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import get
from .models import VerifiedFinding, needs_human_review
from .retrieve import load_goals, load_targets
from .rubric import DIMENSION_KEYS

LEVEL_STATUS = {
    1: "general disclosure only",
    2: "partial disclosure",
    3: "quantified disclosure",
    4: "quantified disclosure",
}

NO_DISCLOSURE = "no scored disclosure identified"

DIMENSION_LABELS = {
    "specificity": "Specificity",
    "measurable_indicator": "Measurable indicator",
    "deadline": "Deadline",
    "baseline": "Baseline",
    "reported_progress": "Reported progress",
}

PRESENT = "present"
ABSENT = "absent"
NOT_ASSESSABLE = "not_assessable"

FLAG_TEXT = {
    "low_specificity": "low specificity",
    "disclosure_gap": "disclosure gap",
    "no_measurable_indicator": "disclosure gap: no measurable indicator",
    "no_deadline": "disclosure gap: no deadline",
    "no_baseline": "disclosure gap: no baseline",
    "no_reported_progress": "disclosure gap: no reported progress",
    "not_assessable": "not assessable: needs human review",
}


@dataclass
class TargetCoverage:
    """Coverage for one SDG target within a focus goal."""

    target_id: str
    target_goal: int
    target_text: str
    finding_count: int = 0
    best_level: int | None = None
    pages: list[int] = field(default_factory=list)
    status: str = NO_DISCLOSURE

    @property
    def has_disclosure(self) -> bool:
        return self.finding_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_goal": self.target_goal,
            "target_text": self.target_text,
            "finding_count": self.finding_count,
            "best_level": self.best_level,
            "pages": list(self.pages),
            "status": self.status,
        }


@dataclass
class GoalCoverage:
    """Coverage for one SDG goal."""

    goal: int
    short_title: str
    goal_title: str
    targets_total: int
    targets: list[TargetCoverage] = field(default_factory=list)
    finding_count: int = 0
    level_counts: dict[int, int] = field(default_factory=dict)
    best_level: int | None = None
    status: str = NO_DISCLOSURE

    @property
    def covered_targets(self) -> list[TargetCoverage]:
        return [target for target in self.targets if target.has_disclosure]

    @property
    def uncovered_targets(self) -> list[TargetCoverage]:
        return [target for target in self.targets if not target.has_disclosure]

    @property
    def covered_target_count(self) -> int:
        return len(self.covered_targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "short_title": self.short_title,
            "goal_title": self.goal_title,
            "targets_total": self.targets_total,
            "covered_target_count": self.covered_target_count,
            "finding_count": self.finding_count,
            "level_counts": {str(k): v for k, v in self.level_counts.items()},
            "best_level": self.best_level,
            "status": self.status,
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass
class CoverageSummary:
    """Coverage across every focus goal."""

    goals: list[GoalCoverage] = field(default_factory=list)

    @property
    def goals_with_findings(self) -> list[GoalCoverage]:
        return [goal for goal in self.goals if goal.finding_count > 0]

    @property
    def targets_total(self) -> int:
        return sum(goal.targets_total for goal in self.goals)

    @property
    def targets_with_findings(self) -> int:
        return sum(goal.covered_target_count for goal in self.goals)

    def level_counts(self) -> dict[int, int]:
        counts = {level: 0 for level in (1, 2, 3, 4)}
        for goal in self.goals:
            for level, count in goal.level_counts.items():
                counts[level] = counts.get(level, 0) + count
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": [goal.to_dict() for goal in self.goals],
            "goals_total": len(self.goals),
            "goals_with_findings": len(self.goals_with_findings),
            "targets_total": self.targets_total,
            "targets_with_findings": self.targets_with_findings,
            "level_counts": {
                str(k): v for k, v in self.level_counts().items()
            },
        }


def target_sort_key(target_id: str) -> tuple[tuple[int, Any], ...]:
    """Sort 12.2 before 12.10, and lettered targets after numbered ones."""
    parts = [part for part in re.split(r"([0-9]+)", target_id) if part]
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in parts
    )


def _status_for_level(level: int | None) -> str:
    if level is None:
        return NO_DISCLOSURE
    return LEVEL_STATUS.get(int(level), NO_DISCLOSURE)


def _focus_goals(
    findings: Sequence[VerifiedFinding],
    config: dict[str, Any],
    focus_goals: Sequence[int] | None,
) -> list[int]:
    if focus_goals:
        return sorted({int(goal) for goal in focus_goals})
    configured = get(config, "report", "focus_sdgs", default=None)
    if configured:
        return sorted({int(goal) for goal in configured})
    return sorted({finding.judged.sdg_goal for finding in findings})


@dataclass
class ClaimCoverage:
    """The five-dimension evidence matrix for one claim (finding)."""

    passage_id: str
    target_id: str
    goal: int
    page: int
    level: int | None
    scored: bool
    quote_verified: bool
    semantic_support: str
    dimensions: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    quote: str = ""
    claim: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "target_id": self.target_id,
            "goal": self.goal,
            "page": self.page,
            "level": self.level,
            "scored": self.scored,
            "quote_verified": self.quote_verified,
            "semantic_support": self.semantic_support,
            "dimensions": dict(self.dimensions),
            "flags": list(self.flags),
            "flag_text": [FLAG_TEXT.get(flag, flag) for flag in self.flags],
            "quote": self.quote,
            "claim": self.claim,
        }


@dataclass
class CoverageMatrix:
    """Evidence coverage matrix across all reviewed claims."""

    claims: list[ClaimCoverage] = field(default_factory=list)

    def dimension_counts(self) -> dict[str, dict[str, int]]:
        counts = {
            key: {PRESENT: 0, ABSENT: 0, NOT_ASSESSABLE: 0}
            for key in DIMENSION_KEYS
        }
        for claim in self.claims:
            for key in DIMENSION_KEYS:
                status = claim.dimensions.get(key, NOT_ASSESSABLE)
                counts[key][status] = counts[key].get(status, 0) + 1
        return counts

    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.claims:
            for flag in claim.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "dimension_counts": self.dimension_counts(),
            "flag_counts": self.flag_counts(),
            "dimension_labels": dict(DIMENSION_LABELS),
            "flag_text": dict(FLAG_TEXT),
        }


def _claim_flags(dimensions: dict[str, str], scored: bool) -> list[str]:
    if not scored:
        return ["not_assessable"]
    absent = [
        key for key in DIMENSION_KEYS if dimensions.get(key) == ABSENT
    ]
    if not absent:
        return []
    flags = ["disclosure_gap"]
    for key in absent:
        if key == "specificity":
            flags.append("low_specificity")
        else:
            flags.append(f"no_{key}")
    return flags


def build_coverage_matrix(
    findings: Sequence[VerifiedFinding],
) -> CoverageMatrix:
    """Build the evidence coverage matrix for every reviewed claim.

    ``present`` means the judge identified the dimension in the verified quote.
    ``absent`` is a disclosure gap in that quote, not proof that the company
    does nothing. ``not_assessable`` covers findings that could not be scored
    because the quote was unverified, the judge output was malformed, or the
    semantic-support check was uncertain.
    """
    matrix = CoverageMatrix()
    for finding in findings:
        if not finding.judged.relevant and not needs_human_review(finding):
            # Clean "not relevant" candidates are not claims about the target.
            continue

        scored = finding.is_scored
        if scored:
            dimensions = {
                key: (
                    PRESENT
                    if bool((finding.judged.dimensions or {}).get(key, False))
                    else ABSENT
                )
                for key in DIMENSION_KEYS
            }
        else:
            dimensions = {key: NOT_ASSESSABLE for key in DIMENSION_KEYS}

        matrix.claims.append(
            ClaimCoverage(
                passage_id=finding.judged.candidate.passage_id,
                target_id=finding.judged.sdg_target,
                goal=finding.judged.sdg_goal,
                page=finding.source_page,
                level=finding.level,
                scored=scored,
                quote_verified=finding.quote_verified,
                semantic_support=finding.semantic_support,
                dimensions=dimensions,
                flags=_claim_flags(dimensions, scored),
                quote=finding.judged.supporting_quote,
                claim=finding.judged.claim,
            )
        )

    matrix.claims.sort(
        key=lambda claim: (claim.goal, target_sort_key(claim.target_id), claim.page)
    )
    return matrix


def build_coverage(
    findings: Sequence[VerifiedFinding],
    config: dict[str, Any],
    focus_goals: Sequence[int] | None = None,
) -> CoverageSummary:
    """Build per-goal and per-target coverage from scored findings only."""
    goals = _focus_goals(findings, config, focus_goals)
    if not goals:
        return CoverageSummary()

    goal_meta = {int(item["goal"]): item for item in load_goals(config, goals)}
    targets = load_targets(config, goals)

    # Group scored findings by target, keeping only scored entries.
    by_target: dict[str, list[VerifiedFinding]] = {}
    for finding in findings:
        if not finding.is_scored:
            continue
        by_target.setdefault(finding.judged.sdg_target, []).append(finding)

    target_entries: dict[str, TargetCoverage] = {}
    for target in targets:
        target_id = str(target["id"])
        entry = TargetCoverage(
            target_id=target_id,
            target_goal=int(target["goal"]),
            target_text=str(target["text"]),
        )
        target_entries[target_id] = entry

    # Defensive: keep a target the judge returned even if it is missing from
    # the focus target list, so a scored finding is never hidden by metadata.
    for target_id, target_findings in by_target.items():
        if target_id in target_entries:
            continue
        sample = target_findings[0].judged
        target_entries[target_id] = TargetCoverage(
            target_id=target_id,
            target_goal=int(sample.sdg_goal),
            target_text=str(sample.candidate.target_text or sample.sdg_target),
        )

    for target_id, target_findings in by_target.items():
        entry = target_entries[target_id]
        levels = [
            int(finding.level)
            for finding in target_findings
            if finding.level is not None
        ]
        entry.finding_count = len(target_findings)
        entry.best_level = max(levels) if levels else None
        entry.pages = sorted({finding.source_page for finding in target_findings})
        entry.status = _status_for_level(entry.best_level)

    summary = CoverageSummary()
    for goal in goals:
        meta = goal_meta.get(goal, {})
        goal_targets = sorted(
            (
                entry
                for entry in target_entries.values()
                if entry.target_goal == goal
            ),
            key=lambda entry: target_sort_key(entry.target_id),
        )
        level_counts = {level: 0 for level in (1, 2, 3, 4)}
        for target in goal_targets:
            for finding in by_target.get(target.target_id, []):
                if finding.level is not None:
                    level_counts[int(finding.level)] += 1
        best_level = max(
            (target.best_level for target in goal_targets if target.best_level),
            default=None,
        )
        summary.goals.append(
            GoalCoverage(
                goal=goal,
                short_title=str(meta.get("short_title", f"SDG {goal}")),
                goal_title=str(meta.get("title", "")),
                targets_total=len(goal_targets),
                targets=goal_targets,
                finding_count=sum(target.finding_count for target in goal_targets),
                level_counts=level_counts,
                best_level=best_level,
                status=_status_for_level(best_level),
            )
        )
    return summary
