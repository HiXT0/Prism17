"""Descriptive cross-company SDG disclosure comparison.

The comparison is deliberately descriptive, not a ranking. It counts scored
findings, targets with scored disclosure, rubric-level distribution, and the
number of findings at or above a configurable level. It does not normalize for
report length, page coverage, sector, or retrieval settings, and it must not be
used to infer real-world performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .audit_io import findings_from_payload, infer_reporting_year, load_payload
from .config import (
    coerce_rubric_level,
    display_path,
    get,
    load_config,
    resolve_path,
)
from .models import VerifiedFinding, is_not_relevant, needs_human_review

LOGGER = logging.getLogger(__name__)

LEVELS = (1, 2, 3, 4)


def _level_counts(findings: Sequence[VerifiedFinding]) -> dict[int, int]:
    counts = {level: 0 for level in LEVELS}
    for finding in findings:
        if finding.level is not None and int(finding.level) in counts:
            counts[int(finding.level)] += 1
    return counts


@dataclass
class GoalProfile:
    """Descriptive metrics for one SDG goal and one company."""

    goal: int
    scored_findings: int = 0
    targets_with_scored_disclosure: int = 0
    best_level: int | None = None
    level_counts: dict[int, int] = field(
        default_factory=lambda: {level: 0 for level in LEVELS}
    )
    high_level_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "scored_findings": self.scored_findings,
            "targets_with_scored_disclosure": self.targets_with_scored_disclosure,
            "best_level": self.best_level,
            "level_counts": {str(k): v for k, v in self.level_counts.items()},
            "high_level_count": self.high_level_count,
        }


@dataclass
class CompanyProfile:
    """Descriptive metrics for one company audit."""

    label: str
    path: str
    title: str
    year: int | None
    findings_total: int
    scored_total: int
    high_level_count: int
    best_level: int | None
    level_counts: dict[int, int]
    review_counts: dict[str, int]
    goals: list[GoalProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "title": self.title,
            "year": self.year,
            "findings_total": self.findings_total,
            "scored_total": self.scored_total,
            "high_level_count": self.high_level_count,
            "best_level": self.best_level,
            "level_counts": {str(k): v for k, v in self.level_counts.items()},
            "review_counts": dict(self.review_counts),
            "goals": [goal.to_dict() for goal in self.goals],
        }


@dataclass
class ComparisonReport:
    """A descriptive comparison across companies."""

    companies: list[CompanyProfile] = field(default_factory=list)
    goals: list[int] = field(default_factory=list)
    high_level_label: str = "L3+"
    min_level: int = 3
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "high_level_label": self.high_level_label,
            "min_level": self.min_level,
            "goals": list(self.goals),
            "companies": [company.to_dict() for company in self.companies],
            "notes": list(self.notes),
        }


def build_company_profile(
    payload: dict[str, Any],
    *,
    label: str,
    path: str = "",
    min_level: int = 3,
    focus_goals: Sequence[int] | None = None,
) -> CompanyProfile:
    """Build descriptive metrics for one audit payload."""
    level = coerce_rubric_level(min_level, setting="compare min_level")
    findings = findings_from_payload(payload)
    document = payload.get("document")
    title = (
        str(document.get("title") or "")
        if isinstance(document, dict)
        else ""
    ) or (Path(path).stem if path else label)

    scored = [finding for finding in findings if finding.is_scored]
    high_level = [
        finding
        for finding in scored
        if finding.level is not None and int(finding.level) >= level
    ]
    goals_present = sorted({int(finding.judged.sdg_goal) for finding in scored})
    selected_goals = (
        sorted({int(goal) for goal in focus_goals})
        if focus_goals
        else goals_present
    )

    goal_profiles: list[GoalProfile] = []
    for goal in selected_goals:
        goal_findings = [
            finding
            for finding in scored
            if int(finding.judged.sdg_goal) == goal
        ]
        levels = [
            int(finding.level)
            for finding in goal_findings
            if finding.level is not None
        ]
        goal_profiles.append(
            GoalProfile(
                goal=goal,
                scored_findings=len(goal_findings),
                targets_with_scored_disclosure=len(
                    {
                        str(finding.judged.sdg_target or "").strip()
                        for finding in goal_findings
                        if str(finding.judged.sdg_target or "").strip()
                    }
                ),
                best_level=max(levels) if levels else None,
                level_counts=_level_counts(goal_findings),
                high_level_count=sum(
                    1
                    for finding in goal_findings
                    if finding.level is not None
                    and int(finding.level) >= level
                ),
            )
        )

    return CompanyProfile(
        label=label,
        path=path,
        title=title,
        year=infer_reporting_year(payload),
        findings_total=len(findings),
        scored_total=len(scored),
        high_level_count=len(high_level),
        best_level=(
            max(int(finding.level) for finding in scored if finding.level)
            if scored
            else None
        ),
        level_counts=_level_counts(scored),
        review_counts={
            "scored": len(scored),
            "not_relevant": sum(
                1 for finding in findings if is_not_relevant(finding)
            ),
            "needs_review": sum(
                1 for finding in findings if needs_human_review(finding)
            ),
            "unscored": len(findings) - len(scored),
        },
        goals=goal_profiles,
    )


def compare_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    paths: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    min_level: int | None = None,
    focus_goals: Sequence[int] | None = None,
) -> ComparisonReport:
    """Compare several audit payloads descriptively."""
    cfg = config or {}
    compare_cfg = get(cfg, "compare", default={}) or {}
    level = coerce_rubric_level(
        min_level
        if min_level is not None
        else compare_cfg.get("min_level", 3),
        setting="compare min_level",
    )
    configured_label = str(compare_cfg.get("high_level_label", "L3+"))
    high_level_label = (
        configured_label
        if int(compare_cfg.get("min_level", 3)) == level
        else f"L{level}+"
    )

    if focus_goals:
        selected_goals = sorted({int(goal) for goal in focus_goals})
    else:
        default_sdg = compare_cfg.get("default_sdg")
        selected_goals = [int(default_sdg)] if default_sdg else []

    resolved_labels = list(labels or [])
    resolved_paths = list(paths or [])
    if resolved_labels and len(resolved_labels) != len(payloads):
        raise ValueError("labels must match the number of audit files")
    if resolved_paths and len(resolved_paths) != len(payloads):
        raise ValueError("paths must match the number of audit files")

    companies: list[CompanyProfile] = []
    for index, payload in enumerate(payloads):
        path = resolved_paths[index] if resolved_paths else ""
        label = (
            resolved_labels[index]
            if resolved_labels
            else (Path(path).stem if path else f"company {index + 1}")
        )
        companies.append(
            build_company_profile(
                payload,
                label=label,
                path=display_path(path, cfg) if path else "",
                min_level=level,
                focus_goals=selected_goals or None,
            )
        )

    report = ComparisonReport(
        companies=companies,
        goals=selected_goals,
        high_level_label=high_level_label,
        min_level=level,
    )
    report.notes = [
        "Descriptive comparison only; this is not a ranking.",
        "Scored findings require a verified quote, a page number, and a rubric "
        "level.",
        f"{high_level_label} means rubric Level {level} or above.",
        "Metrics are not normalized for report length, page coverage, sector, "
        "or retrieval settings.",
        "A lower count may reflect retrieval or reporting choices, not a "
        "weaker real-world commitment.",
    ]
    return report


def _format_company_row(company: CompanyProfile, report: ComparisonReport) -> str:
    goals_with_findings = sum(
        1 for goal in company.goals if goal.scored_findings > 0
    )
    return (
        f"| {company.label} | {company.year or '-'} | "
        f"{company.findings_total} | {company.scored_total} | "
        f"{company.high_level_count} | {company.best_level or '-'} | "
        f"{goals_with_findings}/{len(company.goals)} |"
    )


def render_markdown(report: ComparisonReport) -> str:
    """Render a neutral Markdown comparison report."""
    lines = [
        "# Cross-company SDG comparison",
        "",
        "This is a descriptive comparison of disclosure, not a ranking or an "
        "assessment of real-world performance.",
        "",
        f"- High-level threshold: {report.high_level_label} "
        f"(Level {report.min_level} or above)",
        "",
        "## Company summary",
        "",
        "| Company | Year | Findings reviewed | Scored | "
        f"{report.high_level_label} | Best level | Goals with scored disclosure |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        _format_company_row(company, report) for company in report.companies
    )
    lines.append("")

    if report.goals:
        lines.extend(
            [
                "## Per-goal metrics",
                "",
                "| Company | SDG | Scored findings | Targets with scored "
                f"disclosure | Best level | {report.high_level_label} |",
                "|---|---|---|---|---|---|",
            ]
        )
        for company in report.companies:
            for goal in company.goals:
                lines.append(
                    f"| {company.label} | {goal.goal} | "
                    f"{goal.scored_findings} | "
                    f"{goal.targets_with_scored_disclosure} | "
                    f"{goal.best_level or '-'} | "
                    f"{goal.high_level_count} |"
                )
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def _parse_goals(values: Sequence[int] | None) -> list[int] | None:
    if not values:
        return None
    goals: list[int] = []
    for value in values:
        goal = int(value)
        if not 1 <= goal <= 17:
            raise ValueError(f"SDG goal out of range: {goal}")
        goals.append(goal)
    return goals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Describe SDG disclosure metrics across audit JSON files."
    )
    parser.add_argument("audits", nargs="+", help="Audit JSON files")
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Company labels"
    )
    parser.add_argument(
        "--sdg",
        nargs="+",
        type=int,
        default=None,
        help="Restrict per-goal metrics to these SDG goals",
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=None,
        help="High-level rubric threshold (default: config compare.min_level)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write comparison JSON here")
    parser.add_argument(
        "--markdown", default=None, help="Write comparison Markdown here"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        if args.labels and len(args.labels) != len(args.audits):
            raise ValueError("--labels must match the number of audit files")
        payloads = [load_payload(path) for path in args.audits]
        report = compare_payloads(
            payloads,
            labels=args.labels,
            paths=args.audits,
            config=config,
            min_level=args.min_level,
            focus_goals=_parse_goals(args.sdg),
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2

    markdown = render_markdown(report)
    print(markdown)
    if args.out:
        path = resolve_path(args.out, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote comparison JSON: {path}")
    if args.markdown:
        path = resolve_path(args.markdown, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"Wrote comparison Markdown: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
