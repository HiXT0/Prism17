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
from .models import VerifiedFinding
from .retrieve import load_goals, load_targets

LEVEL_STATUS = {
    1: "general disclosure only",
    2: "partial disclosure",
    3: "quantified disclosure",
    4: "quantified disclosure",
}

NO_DISCLOSURE = "no scored disclosure identified"

# The five rubric dimensions, in report order. The judge returns these as
# booleans; coverage only counts the flags already recorded on scored findings.
EVIDENCE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("specificity", "Specificity"),
    ("measurable_indicator", "Measurable indicator"),
    ("deadline", "Deadline"),
    ("baseline", "Baseline"),
    ("reported_progress", "Reported progress"),
)

DIMENSION_KEYS = tuple(key for key, _ in EVIDENCE_DIMENSIONS)
DIMENSION_LABELS = dict(EVIDENCE_DIMENSIONS)


@dataclass
class DimensionCoverage:
    """How many scored findings disclose one rubric dimension."""

    key: str
    label: str
    disclosed: int = 0
    findings: int = 0

    @property
    def share(self) -> float | None:
        """Share of scored findings with this dimension, or None if empty."""
        if self.findings <= 0:
            return None
        return self.disclosed / self.findings

    def text(self) -> str:
        """Compact cell text for reports, with an ASCII dash when empty."""
        share = self.share
        if share is None:
            return "-"
        return f"{self.disclosed}/{self.findings} ({share:.0%})"

    def to_dict(self) -> dict[str, Any]:
        share = self.share
        return {
            "key": self.key,
            "label": self.label,
            "disclosed": self.disclosed,
            "findings": self.findings,
            "share": None if share is None else round(share, 4),
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
    dimensions: list[DimensionCoverage] = field(default_factory=list)
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
            "dimensions": [item.to_dict() for item in self.dimensions],
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

    @property
    def dimension_totals(self) -> list[DimensionCoverage]:
        """Aggregate the evidence matrix across every focus goal."""
        totals = {
            key: DimensionCoverage(key=key, label=label)
            for key, label in EVIDENCE_DIMENSIONS
        }
        for goal in self.goals:
            for dimension in goal.dimensions:
                bucket = totals[dimension.key]
                bucket.disclosed += dimension.disclosed
                bucket.findings += dimension.findings
        return [totals[key] for key, _ in EVIDENCE_DIMENSIONS]

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
            "dimension_totals": [
                item.to_dict() for item in self.dimension_totals
            ],
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
        goal_findings = [
            finding
            for target in goal_targets
            for finding in by_target.get(target.target_id, [])
        ]
        dimension_counts = {key: 0 for key in DIMENSION_KEYS}
        for finding in goal_findings:
            flags = finding.judged.dimensions or {}
            for key in DIMENSION_KEYS:
                if flags.get(key):
                    dimension_counts[key] += 1
        dimensions = [
            DimensionCoverage(
                key=key,
                label=label,
                disclosed=dimension_counts[key],
                findings=len(goal_findings),
            )
            for key, label in EVIDENCE_DIMENSIONS
        ]
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
                dimensions=dimensions,
                best_level=best_level,
                status=_status_for_level(best_level),
            )
        )
    return summary
