"""Cross-company comparison of SDG disclosure.

The comparison is deliberately descriptive. It puts the same disclosure
measures side by side; it does not rank companies and it does not compare
real-world performance. Every measure counts only findings that are scored
(verified quote, page number, rubric level, and Stage 2 support not blocking).

Different reports can cover different pages, goals, and candidate budgets, so a
difference in the table can reflect retrieval and judging scope rather than the
company's disclosure. The rendered report states this caveat.

Run:
    python -m prism17.compare outputs/a.json outputs/b.json \
        --out outputs/compare.json --markdown outputs/compare.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import get, load_config, resolve_path
from .coverage import target_sort_key
from .models import VerifiedFinding, is_not_relevant, needs_human_review
from .tracking import infer_reporting_year

LOGGER = logging.getLogger(__name__)

LEVELS = (1, 2, 3, 4)


@dataclass
class GoalProfile:
    """Scored disclosure for one goal within one company audit."""

    goal: int
    scored_findings: int = 0
    level_counts: dict[int, int] = field(
        default_factory=lambda: {level: 0 for level in LEVELS}
    )
    targets: list[str] = field(default_factory=list)
    min_level: int = 3

    @property
    def targets_with_scored(self) -> int:
        return len(self.targets)

    @property
    def best_level(self) -> int | None:
        levels = [
            level for level, count in self.level_counts.items() if count > 0
        ]
        return max(levels) if levels else None

    @property
    def high_level_count(self) -> int:
        return sum(
            self.level_counts.get(level, 0)
            for level in LEVELS
            if level >= self.min_level
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "scored_findings": self.scored_findings,
            "level_counts": {
                str(level): self.level_counts.get(level, 0) for level in LEVELS
            },
            "targets_with_scored": self.targets_with_scored,
            "targets": list(self.targets),
            "best_level": self.best_level,
            "min_level": self.min_level,
            "high_level_count": self.high_level_count,
        }


@dataclass
class CompanyProfile:
    """Disclosure profile for one company audit."""

    label: str
    reporting_year: int | None = None
    model: str = ""
    pages: int = 0
    findings_total: int = 0
    scored_total: int = 0
    level_counts: dict[int, int] = field(
        default_factory=lambda: {level: 0 for level in LEVELS}
    )
    needs_review_total: int = 0
    not_relevant_total: int = 0
    goals: dict[int, GoalProfile] = field(default_factory=dict)
    min_level: int = 3

    @property
    def best_level(self) -> int | None:
        levels = [
            level for level, count in self.level_counts.items() if count > 0
        ]
        return max(levels) if levels else None

    @property
    def high_level_count(self) -> int:
        return sum(
            self.level_counts.get(level, 0)
            for level in LEVELS
            if level >= self.min_level
        )

    @property
    def goals_with_scored(self) -> list[int]:
        return sorted(
            goal for goal, profile in self.goals.items()
            if profile.scored_findings > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reporting_year": self.reporting_year,
            "model": self.model,
            "pages": self.pages,
            "findings_total": self.findings_total,
            "scored_total": self.scored_total,
            "level_counts": {
                str(level): self.level_counts.get(level, 0) for level in LEVELS
            },
            "best_level": self.best_level,
            "min_level": self.min_level,
            "high_level_count": self.high_level_count,
            "needs_review_total": self.needs_review_total,
            "not_relevant_total": self.not_relevant_total,
            "goals_with_scored": self.goals_with_scored,
            "goals": [
                profile.to_dict()
                for _, profile in sorted(self.goals.items())
            ],
        }


@dataclass
class ComparisonReport:
    """Descriptive comparison across two or more company audits."""

    profiles: list[CompanyProfile] = field(default_factory=list)
    goals: list[int] = field(default_factory=list)
    min_level: int = 3
    high_level_label: str = "L3+"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "min_level": self.min_level,
            "high_level_label": self.high_level_label,
            "goals": list(self.goals),
            "companies": [profile.to_dict() for profile in self.profiles],
            "notes": list(self.notes),
        }


def _findings_from_payload(payload: dict[str, Any]) -> list[VerifiedFinding]:
    if "findings" not in payload:
        raise ValueError("audit JSON has no 'findings' list")
    return [
        VerifiedFinding.from_dict(item)
        for item in (payload.get("findings") or [])
    ]


def _label(document: dict[str, Any], fallback: str) -> str:
    title = str(document.get("title") or "").strip()
    if title:
        return title
    pdf_path = str(document.get("pdf_path") or "").strip()
    return Path(pdf_path).stem if pdf_path else fallback


def _unique_label(label: str, used: set[str]) -> str:
    if label not in used:
        used.add(label)
        return label
    index = 2
    while f"{label} ({index})" in used:
        index += 1
    unique = f"{label} ({index})"
    used.add(unique)
    return unique


def build_company_profile(
    payload: dict[str, Any],
    *,
    label: str | None = None,
    fallback_label: str = "company",
    goals: Sequence[int] | None = None,
    min_level: int = 3,
) -> CompanyProfile:
    """Summarise one audit payload without ranking it against another."""
    document = payload.get("document") or {}
    settings = payload.get("settings") or {}
    stats = payload.get("stats") or {}
    findings = _findings_from_payload(payload)
    goal_filter = {int(goal) for goal in goals} if goals else None

    profile = CompanyProfile(
        label=label or _label(document, fallback_label),
        reporting_year=infer_reporting_year(document),
        model=str(settings.get("model") or ""),
        pages=int(
            document.get("page_count") or stats.get("pages") or 0
        ),
        findings_total=len(findings),
        min_level=min_level,
    )

    for finding in findings:
        if needs_human_review(finding):
            profile.needs_review_total += 1
        elif is_not_relevant(finding):
            profile.not_relevant_total += 1

        if not finding.is_scored:
            continue
        goal = int(finding.judged.sdg_goal)
        if goal_filter is not None and goal not in goal_filter:
            continue
        level = int(finding.level)
        if level not in profile.level_counts:
            continue

        profile.scored_total += 1
        profile.level_counts[level] += 1
        goal_profile = profile.goals.setdefault(
            goal, GoalProfile(goal=goal, min_level=min_level)
        )
        goal_profile.scored_findings += 1
        goal_profile.level_counts[level] += 1
        target_id = str(finding.judged.sdg_target)
        if target_id and target_id not in goal_profile.targets:
            goal_profile.targets.append(target_id)

    for goal_profile in profile.goals.values():
        goal_profile.targets.sort(key=target_sort_key)
    return profile


def compare_payloads(
    payloads: Sequence[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    goals: Sequence[int] | None = None,
    min_level: int | None = None,
) -> ComparisonReport:
    """Build a descriptive comparison from audit JSON payloads."""
    if not payloads:
        raise ValueError("at least one audit payload is required")
    if labels is not None and len(labels) != len(payloads):
        raise ValueError("the number of labels must match the number of audits")

    cfg = config or {}
    compare_cfg = get(cfg, "compare", default={}) or {}
    configured_min = int(compare_cfg.get("min_level", 3))
    level = int(min_level if min_level is not None else configured_min)
    if level not in LEVELS:
        raise ValueError(
            f"compare min_level must be one of {LEVELS}, got {level}"
        )
    if level == configured_min and compare_cfg.get("high_level_label"):
        level_label = str(compare_cfg["high_level_label"])
    else:
        level_label = f"L{level}+"
    report = ComparisonReport(
        min_level=level,
        high_level_label=level_label,
    )

    goal_filter = {int(goal) for goal in goals} if goals else None
    used: set[str] = set()
    for index, payload in enumerate(payloads, start=1):
        raw_label = (
            labels[index - 1]
            if labels is not None
            else None
        )
        profile = build_company_profile(
            payload,
            label=raw_label,
            fallback_label=f"company {index}",
            goals=goals,
            min_level=level,
        )
        profile.label = _unique_label(profile.label, used)
        report.profiles.append(profile)

    if goal_filter is not None:
        report.goals = sorted(goal_filter)
    else:
        report.goals = sorted(
            {goal for profile in report.profiles for goal in profile.goals}
        )

    report.notes = [
        "The comparison describes disclosure only; it does not assess "
        "real-world performance and is not a ranking.",
        "Only scored findings (verified quote, page number, rubric level, and "
        "no blocking semantic-support result) are counted.",
        f"{report.high_level_label} means rubric Level {level} or above.",
        "Reports can differ in page coverage, focus goals, retrieval depth, and "
        "the number of judged candidates; a difference in the table may reflect "
        "those run settings rather than the company's disclosure.",
        "Read each row with the source audit, verified quotes, and page numbers.",
    ]
    if goal_filter is not None:
        report.notes.append(
            "Scored and per-goal metrics are restricted to the selected goals; "
            "the review-status counts cover the whole audit."
        )
    return report


def _format_level(level: int | None) -> str:
    return str(level) if level is not None else "n/a"


def _company_row(profile: CompanyProfile) -> str:
    goals = ", ".join(str(goal) for goal in profile.goals_with_scored) or "n/a"
    year = str(profile.reporting_year) if profile.reporting_year else "n/a"
    return (
        f"| {profile.label} | {year} | {profile.findings_total} | "
        f"{profile.scored_total} | {profile.high_level_count} | "
        f"{_format_level(profile.best_level)} | {goals} |"
    )


def render_markdown(report: ComparisonReport) -> str:
    """Render a neutral cross-company comparison."""
    lines = [
        "# Cross-company SDG disclosure comparison",
        "",
        "> This compares **disclosure** across audits. It does not verify "
        "real-world performance, and it is not a ranking of the companies.",
        "",
        "## Summary",
        "",
        "| Company | Year | Findings reviewed | Scored | "
        f"{report.high_level_label} | Best level | Goals with scored disclosure |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile in report.profiles:
        lines.append(_company_row(profile))
    lines.append("")

    lines.extend(
        [
            "## Scored disclosure by SDG goal",
            "",
            "| Company | SDG | Scored findings | Targets with scored disclosure | "
            f"Best level | {report.high_level_label} |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if not report.goals:
        lines.append("| n/a | n/a | 0 | 0 | n/a | 0 |")
    else:
        for profile in report.profiles:
            for goal in report.goals:
                goal_profile = profile.goals.get(goal)
                scored = goal_profile.scored_findings if goal_profile else 0
                targets = (
                    goal_profile.targets_with_scored if goal_profile else 0
                )
                best = goal_profile.best_level if goal_profile else None
                high_level_count = (
                    goal_profile.high_level_count if goal_profile else 0
                )
                lines.append(
                    f"| {profile.label} | {goal} | {scored} | {targets} | "
                    f"{_format_level(best)} | {high_level_count} |"
                )
    lines.append("")

    lines.extend(
        [
            "## Level distribution",
            "",
            "| Company | L1 | L2 | L3 | L4 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for profile in report.profiles:
        lines.append(
            "| "
            + profile.label
            + " | "
            + " | ".join(
                str(profile.level_counts.get(level, 0)) for level in LEVELS
            )
            + " |"
        )
    lines.append("")

    lines.extend(
        [
            "## Review status",
            "",
            "| Company | Needs review | Judged not relevant |",
            "| --- | --- | --- |",
        ]
    )
    for profile in report.profiles:
        lines.append(
            f"| {profile.label} | {profile.needs_review_total} | "
            f"{profile.not_relevant_total} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(
    report: ComparisonReport, path: str | Path, config: dict[str, Any]
) -> Path:
    target = resolve_path(path, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SDG disclosure metrics across two or more Prism17 audits."
        )
    )
    parser.add_argument("audits", nargs="+", help="Audit JSON files")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Company labels in the same order as the audit files",
    )
    parser.add_argument(
        "--sdg",
        type=int,
        action="append",
        default=None,
        help="Limit the comparison to a goal (repeatable)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write comparison JSON here")
    parser.add_argument(
        "--markdown", default=None, help="Write a Markdown comparison here"
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=None,
        help="Level counted as high-specificity (default: config compare.min_level)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    # Keep report text printable on a Windows code page that lacks some PDF
    # characters. Files are still written as UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        payloads = [
            json.loads(resolve_path(path, config).read_text(encoding="utf-8"))
            for path in args.audits
        ]
        report = compare_payloads(
            payloads,
            config,
            labels=args.labels,
            goals=args.sdg,
            min_level=args.min_level,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(render_markdown(report))
    if args.out:
        path = _write_json(report, args.out, config)
        print(f"\nWrote comparison JSON: {path}")
    if args.markdown:
        path = resolve_path(args.markdown, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote comparison Markdown: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
