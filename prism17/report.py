"""Phase 3 SDG Alignment Report: coverage, findings, and guidance.

The report is rendered from a completed Phase 2 audit, either in memory or from
the audit JSON. Rendering is deterministic and never calls an LLM. Every scored
finding keeps its page number and verified quote; findings that need human
review are listed separately and are never presented as scored.

Run:
    python -m prism17.report outputs/sjs.phase2.json --out outputs/sjs.report.md
    python -m prism17.report outputs/sjs.phase2.json --html outputs/sjs.report.html
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .config import get, load_config, resolve_path
from .coverage import (
    EVIDENCE_DIMENSIONS,
    CoverageSummary,
    GoalCoverage,
    TargetCoverage,
    build_coverage,
    target_sort_key,
)
from .guidance import (
    COVERAGE_CAVEAT,
    NO_DISCLOSURE_GUIDANCE,
    guidance_for_finding,
    guidance_for_goal,
)
from .models import (
    VerifiedFinding,
    is_not_relevant,
    needs_human_review,
)
from .review import (
    ReviewDecision,
    export_rows,
    finding_key,
    load_reviews,
    summarise_reviews,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineResult

LOGGER = logging.getLogger(__name__)

DEFAULT_TITLE = "SDG Alignment Report"

DISCLAIMER = (
    "This report assesses **disclosure quality**, not real-world "
    "sustainability performance. It is decision support: a human reviewer "
    "should confirm or override every finding before it is used."
)

DISCLAIMER_TEXT = DISCLAIMER.replace("**", "")

METHOD_ITEMS = (
    (
        "Scope",
        "The report describes the presence and specificity of disclosure. It "
        "does not verify real-world impact, performance, or intent.",
    ),
    (
        "Parsing",
        "Text and tables are extracted page by page. Unreadable pages are "
        "recorded and never treated as evidence of no disclosure.",
    ),
    (
        "Retrieval",
        "Passages are compared with the UN SDG targets using a local "
        "sentence-embedding model and FAISS on CPU. Mapping a passage to a "
        "target is an approximation.",
    ),
    (
        "Judging",
        "A local Ollama model returns strict JSON. Malformed output is "
        "retained as needing human review, not scored.",
    ),
    (
        "Verification",
        "Stage 1 checks that each quote exists on the cited page, after "
        "whitespace and hyphenation normalisation, with a fuzzy fallback. "
        "Stage 2 semantic support is optional and off by default; when it is "
        "enabled, an ambiguous result keeps the finding unscored.",
    ),
    (
        "Rubric",
        "Levels 1-4 are assigned by deterministic rules over five disclosure "
        "dimensions. A high level means the disclosure is measurable, "
        "time-bound, baseline-anchored, and reports progress.",
    ),
    (
        "Human review",
        "Unverified quotes, malformed judge output, and ambiguous mappings "
        "require human review before use.",
    ),
)


@dataclass
class ReportData:
    """Everything needed to render one SDG Alignment Report."""

    document: dict[str, Any]
    settings: dict[str, Any]
    stats: dict[str, Any]
    coverage: CoverageSummary
    findings: list[VerifiedFinding]
    focus_goals: list[int] = field(default_factory=list)
    company_name: str | None = None
    report_title: str = DEFAULT_TITLE
    generated_at: str = ""
    include_unscored: bool = True
    tool_version: str = ""
    reviews: dict[str, ReviewDecision] = field(default_factory=dict)

    @property
    def scored(self) -> list[VerifiedFinding]:
        return [finding for finding in self.findings if finding.is_scored]

    @property
    def needs_review(self) -> list[VerifiedFinding]:
        return [
            finding for finding in self.findings if needs_human_review(finding)
        ]

    @property
    def not_relevant(self) -> list[VerifiedFinding]:
        return [
            finding for finding in self.findings if is_not_relevant(finding)
        ]

    @property
    def subject(self) -> str:
        if self.company_name:
            return self.company_name
        title = str(self.document.get("title") or "").strip()
        if title:
            return title
        pdf_path = str(self.document.get("pdf_path") or "").strip()
        return Path(pdf_path).stem if pdf_path else "Report"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _resolve_focus_goals(
    config: dict[str, Any],
    findings: Sequence[VerifiedFinding],
    focus_goals: Sequence[int] | None = None,
) -> list[int]:
    if focus_goals:
        return sorted({int(goal) for goal in focus_goals})
    configured = get(config, "report", "focus_sdgs", default=None)
    if configured:
        return sorted({int(goal) for goal in configured})
    return sorted({finding.judged.sdg_goal for finding in findings})


def build_report_data(
    result: "PipelineResult",
    config: dict[str, Any] | None = None,
    *,
    reviews: dict[str, ReviewDecision] | None = None,
) -> ReportData:
    """Build report data from an in-memory pipeline result."""
    cfg = config or result.config or load_config()
    findings = list(result.findings)
    focus_goals = _resolve_focus_goals(cfg, result.judged)
    return ReportData(
        document={
            "pdf_path": result.document.pdf_path,
            "title": result.document.title,
            "page_count": result.document.page_count,
            "sha256": result.document.metadata.get("sha256"),
        },
        settings={
            "model": get(cfg, "llm", "model", default=None),
            "embedding_model": get(
                cfg, "retrieval", "embedding_model", default=None
            ),
            "focus_sdgs": focus_goals,
        },
        stats=result.stats(),
        coverage=build_coverage(result.scored, cfg, focus_goals),
        findings=findings,
        focus_goals=focus_goals,
        company_name=get(cfg, "report", "company_name", default=None),
        report_title=str(get(cfg, "report", "report_title", default=DEFAULT_TITLE)),
        generated_at=_timestamp(),
        include_unscored=bool(
            get(cfg, "report", "include_unscored", default=True)
        ),
        tool_version=str(get(cfg, "project", "version", default="")),
        reviews=dict(reviews or {}),
    )


def report_data_from_json(
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    reviews: dict[str, ReviewDecision] | None = None,
) -> ReportData:
    """Rebuild report data from a Phase 2 audit JSON payload."""
    cfg = config or load_config()
    settings = payload.get("settings") or {}
    document = payload.get("document") or {}
    findings = [
        VerifiedFinding.from_dict(item)
        for item in (payload.get("findings") or [])
    ]
    focus_goals = _resolve_focus_goals(
        cfg,
        findings,
        settings.get("focus_sdgs"),
    )
    return ReportData(
        document=document,
        settings=settings,
        stats=payload.get("stats") or {},
        coverage=build_coverage(
            [finding for finding in findings if finding.is_scored],
            cfg,
            focus_goals,
        ),
        findings=findings,
        focus_goals=focus_goals,
        company_name=get(cfg, "report", "company_name", default=None),
        report_title=str(get(cfg, "report", "report_title", default=DEFAULT_TITLE)),
        generated_at=_timestamp(),
        include_unscored=bool(
            get(cfg, "report", "include_unscored", default=True)
        ),
        tool_version=str(get(cfg, "project", "version", default="")),
        reviews=dict(reviews or {}),
    )


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _squash(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _md_cell(text: Any) -> str:
    return _squash(text).replace("|", "\\|")


def _level_text(level: int | None) -> str:
    return f"Level {level}" if level else "Unscored"


def _group_by_goal(
    findings: Sequence[VerifiedFinding],
) -> dict[int, dict[str, list[VerifiedFinding]]]:
    grouped: dict[int, dict[str, list[VerifiedFinding]]] = {}
    for finding in findings:
        grouped.setdefault(finding.judged.sdg_goal, {}).setdefault(
            finding.judged.sdg_target, []
        ).append(finding)
    for targets in grouped.values():
        for target_findings in targets.values():
            target_findings.sort(
                key=lambda item: (
                    item.source_page,
                    item.judged.candidate.passage_id,
                )
            )
    return grouped


def _coverage_indexes(
    coverage: CoverageSummary,
) -> tuple[dict[int, GoalCoverage], dict[tuple[int, str], TargetCoverage]]:
    goals = {goal.goal: goal for goal in coverage.goals}
    targets = {
        (goal.goal, target.target_id): target
        for goal in coverage.goals
        for target in goal.targets
    }
    return goals, targets


def _level_counts_text(stats: dict[str, Any]) -> str:
    counts = stats.get("level_counts") or {}
    return " · ".join(
        f"L{level} {counts.get(str(level), 0)}" for level in (1, 2, 3, 4)
    )


EVIDENCE_MATRIX_NOTE = (
    "Counts show how many scored findings disclose each of the five rubric "
    "dimensions. The flags come from the judge and describe disclosure in this "
    "run, not real-world performance. A low count is a disclosure observation, "
    "not evidence that the company does nothing."
)


def _evidence_matrix_headers() -> tuple[str, ...]:
    return ("Goal", "Scored findings") + tuple(
        label for _, label in EVIDENCE_DIMENSIONS
    )


def _evidence_matrix_rows(
    coverage: CoverageSummary,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = [
        (
            f"Goal {goal.goal}",
            goal.finding_count,
            *(item.text() for item in goal.dimensions),
        )
        for goal in coverage.goals
    ]
    rows.append(
        (
            "All focus goals",
            sum(goal.finding_count for goal in coverage.goals),
            *(item.text() for item in coverage.dimension_totals),
        )
    )
    return rows


def _review_summary_text(summary: Any) -> str:
    if summary.total == 0:
        return "No scored findings"
    if summary.reviewed == 0:
        return f"Not reviewed ({summary.total} pending)"
    return (
        f"{summary.confirmed} confirmed · {summary.overridden} overridden · "
        f"{summary.flagged} flagged · {summary.pending} pending"
    )


def _review_label(decision: ReviewDecision | None) -> str:
    if decision is None or decision.status == "pending":
        return ""
    if decision.status == "overridden":
        label = f"Overridden to Level {decision.level}"
    else:
        label = decision.status.capitalize()
    if decision.reviewer:
        label += f" by {decision.reviewer}"
    if decision.note:
        label += f" — {decision.note}"
    return label


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return lines


def render_markdown(data: ReportData) -> str:
    """Render the full report as Markdown."""
    lines: list[str] = []
    lines.append(f"# {data.report_title} — {data.subject}")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")

    lines.append("## Report details")
    lines.append("")
    document = data.document
    pdf_name = Path(str(document.get("pdf_path") or "")).name
    detail_rows = [
        ("Document", f"`{pdf_name}`" if pdf_name else "Not recorded"),
        ("Pages parsed", data.stats.get("pages", document.get("page_count", "—"))),
        ("Focus goals", ", ".join(str(goal) for goal in data.focus_goals) or "—"),
        ("Local judge", data.settings.get("model") or "Not recorded"),
        (
            "Embedding model",
            data.settings.get("embedding_model") or "Not recorded",
        ),
        ("Generated", data.generated_at),
        (
            "Tool",
            f"Prism17 {data.tool_version}".strip() or "Prism17",
        ),
    ]
    lines.extend(_md_table(("Field", "Value"), detail_rows))
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    stats = data.stats
    review_summary = summarise_reviews(data.scored, data.reviews)
    summary_rows = [
        ("Scored findings", stats.get("scored", len(data.scored))),
        ("Level distribution", _level_counts_text(stats)),
        (
            "Targets with scored disclosure",
            f"{data.coverage.targets_with_findings} of "
            f"{data.coverage.targets_total} focus targets",
        ),
        ("Judged candidates", stats.get("judged", len(data.findings))),
        ("Not relevant", stats.get("not_relevant", len(data.not_relevant))),
        ("Needs human review", stats.get("needs_review", len(data.needs_review))),
        ("Unreadable pages", stats.get("unreadable_pages", 0)),
        ("Human review", _review_summary_text(review_summary)),
    ]
    lines.extend(_md_table(("Measure", "Value"), summary_rows))
    lines.append("")
    lines.append(COVERAGE_CAVEAT)
    lines.append("")

    lines.append("## Coverage by goal")
    lines.append("")
    if data.coverage.goals:
        coverage_rows = [
            (
                f"Goal {goal.goal}",
                goal.short_title,
                f"{goal.covered_target_count} of {goal.targets_total}",
                goal.finding_count,
                _level_text(goal.best_level),
                goal.status,
            )
            for goal in data.coverage.goals
        ]
        lines.extend(
            _md_table(
                (
                    "Goal",
                    "Focus area",
                    "Targets with scored disclosure",
                    "Scored findings",
                    "Best level",
                    "Status",
                ),
                coverage_rows,
            )
        )
    else:
        lines.append("No focus goals were available for coverage accounting.")
    lines.append("")

    lines.append("## Evidence coverage matrix")
    lines.append("")
    lines.append(EVIDENCE_MATRIX_NOTE)
    lines.append("")
    if data.coverage.goals:
        lines.extend(
            _md_table(
                _evidence_matrix_headers(),
                _evidence_matrix_rows(data.coverage),
            )
        )
    else:
        lines.append(
            "No scored findings were available for dimension accounting."
        )
    lines.append("")

    lines.append("## Scored findings")
    lines.append("")
    if not data.scored:
        lines.append(
            "No scored findings. No finding met both quote verification and "
            "rubric requirements."
        )
        lines.append("")
    else:
        goals, targets = _coverage_indexes(data.coverage)
        grouped = _group_by_goal(data.scored)
        for goal_number in sorted(grouped):
            goal_cov = goals.get(goal_number)
            if goal_cov is not None:
                lines.append(
                    f"### Goal {goal_number} — {goal_cov.short_title}"
                )
                lines.append("")
                if goal_cov.goal_title:
                    lines.append(goal_cov.goal_title)
                    lines.append("")
                lines.append(
                    f"**Goal status:** {goal_cov.status} "
                    f"({_plural(goal_cov.finding_count, 'scored finding')}; "
                    f"{goal_cov.covered_target_count} of "
                    f"{goal_cov.targets_total} targets with scored disclosure)."
                )
                lines.append("")
                for text in guidance_for_goal(goal_cov):
                    lines.append(f"**Goal guidance:** {text}")
                    lines.append("")
            else:
                lines.append(f"### Goal {goal_number}")
                lines.append("")

            for target_id in sorted(grouped[goal_number], key=target_sort_key):
                target_cov = targets.get((goal_number, target_id))
                target_findings = grouped[goal_number][target_id]
                best_level = max(
                    (
                        finding.level
                        for finding in target_findings
                        if finding.level is not None
                    ),
                    default=None,
                )
                lines.append(
                    f"#### Target {target_id} — {_level_text(best_level)}"
                )
                lines.append("")
                if target_cov is not None and target_cov.target_text:
                    lines.append(f"*Target:* {_squash(target_cov.target_text)}")
                    lines.append("")

                for index, finding in enumerate(target_findings, start=1):
                    label = (
                        f"Finding {index} — page {finding.source_page}"
                        if len(target_findings) > 1
                        else f"Page {finding.source_page}"
                    )
                    method = finding.quote_method or "unknown method"
                    lines.append(
                        f"**{label} — quote verified "
                        f"({method} {finding.quote_score:.1f})**"
                    )
                    lines.append("")
                    lines.append(f'> "{_squash(finding.judged.supporting_quote)}"')
                    lines.append("")
                    lines.append(
                        f"- **Assessment:** {_squash(finding.level_rationale)}"
                    )
                    gaps = finding.disclosure_gaps
                    if gaps:
                        lines.append("- **Disclosure gaps:**")
                        for gap in gaps:
                            lines.append(f"  - {_squash(gap)}")
                    else:
                        lines.append(
                            "- **Disclosure gaps:** none identified by the rubric."
                        )
                    suggestions = guidance_for_finding(finding)
                    if suggestions:
                        lines.append("- **Suggested improvements:**")
                        for suggestion in suggestions:
                            lines.append(f"  - {_squash(suggestion)}")
                    else:
                        lines.append(
                            "- **Suggested improvements:** none identified by "
                            "the rubric."
                        )
                    review_label = _review_label(
                        data.reviews.get(finding_key(finding))
                    )
                    if review_label:
                        lines.append(f"- **Human review:** {review_label}")
                    lines.append("")

    review_rows = [
        row
        for row in export_rows(data.scored, data.reviews)
        if row["review_status"] != "pending"
    ]
    if review_rows:
        lines.append("## Human review")
        lines.append("")
        lines.extend(
            _md_table(
                (
                    "Target",
                    "Page",
                    "Rubric level",
                    "Status",
                    "Reviewed level",
                    "Reviewer",
                    "Note",
                ),
                [
                    (
                        row["target"],
                        row["page"],
                        _level_text(row["rubric_level"]),
                        row["review_status"],
                        (
                            f"Level {row['review_level']}"
                            if row["review_level"]
                            else "—"
                        ),
                        row["reviewer"],
                        row["note"],
                    )
                    for row in review_rows
                ],
            )
        )
        lines.append("")

    goals_without = [
        goal for goal in data.coverage.goals if goal.finding_count == 0
    ]
    if goals_without:
        lines.append("## Goals without scored findings")
        lines.append("")
        for goal in goals_without:
            lines.append(f"### Goal {goal.goal} — {goal.short_title}")
            lines.append("")
            if goal.goal_title:
                lines.append(goal.goal_title)
                lines.append("")
            lines.append(NO_DISCLOSURE_GUIDANCE)
            lines.append("")

    if data.include_unscored and data.needs_review:
        lines.append("## Needs human review")
        lines.append("")
        for finding in data.needs_review:
            reason = finding.review_reason or "not scored"
            lines.append(
                f"- **Page {finding.source_page}, target "
                f"{finding.judged.sdg_target}:** {_squash(reason)}"
            )
        lines.append("")

    if data.not_relevant:
        lines.append("## Candidates excluded as not relevant")
        lines.append("")
        count = len(data.not_relevant)
        verb = "was" if count == 1 else "were"
        lines.append(
            f"{_plural(count, 'candidate')} {verb} judged not relevant to the "
            f"retrieved target and {verb} not scored. This count describes "
            "retrieval and judging output, not the company."
        )
        lines.append("")

    lines.append("## Method and limitations")
    lines.append("")
    for label, text in METHOD_ITEMS:
        lines.append(f"- **{label}.** {text}")
    unreadable = int(stats.get("unreadable_pages", 0) or 0)
    if unreadable:
        lines.append(
            f"- **Unreadable pages.** {unreadable} page(s) had no usable text "
            "layer and were not assessed."
        )
    lines.append("")

    lines.append("## Responsible use")
    lines.append("")
    lines.append("- A disclosure gap is not proof of poor performance.")
    lines.append(
        "- A high level reflects measurable, time-bound, baseline-anchored "
        "disclosure, not the quality of the underlying activity."
    )
    lines.append(
        "- Quotes are checked against the source page; semantic support and "
        "rubric decisions should be reviewed by a human."
    )
    lines.append(
        "- This report must not be used to make automated accusations about a "
        "company."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML_STYLE = """
:root { color-scheme: light; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 980px; margin: 2rem auto; padding: 0 1.25rem;
  color: #1f2933; line-height: 1.55; background: #ffffff;
}
h1, h2, h3 { line-height: 1.25; color: #102a43; }
h1 { font-size: 1.8rem; margin-bottom: .4rem; }
h2 { margin-top: 2.2rem; border-bottom: 2px solid #d9e2ec; padding-bottom: .3rem; }
h3 { margin-top: 1.6rem; }
p { margin: .5rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .95rem; }
th, td { border: 1px solid #cbd2d9; padding: .45rem .6rem; text-align: left; vertical-align: top; }
th { background: #f0f4f8; }
blockquote {
  margin: .6rem 0; padding: .6rem 1rem; border-left: 4px solid #9fb3c8;
  background: #f5f7fa; color: #243b53;
}
ul { margin: .4rem 0 .8rem 1.2rem; padding: 0; }
li { margin: .25rem 0; }
.disclaimer {
  border: 1px solid #bcccdc; background: #f0f4f8; padding: .8rem 1rem;
  border-radius: 6px;
}
.badge {
  display: inline-block; border-radius: 999px; padding: .1rem .6rem;
  font-size: .85rem; background: #e4e7eb; color: #243b53; font-weight: 600;
}
.level-1 { background: #fff1e6; color: #8a4b08; }
.level-2 { background: #fff7cc; color: #6b5400; }
.level-3 { background: #e3f9e5; color: #1f6b2a; }
.level-4 { background: #d7f0ff; color: #0b4f6c; }
.finding { border-top: 1px solid #e4e7eb; padding-top: .8rem; margin-top: .8rem; }
.meta { color: #52606d; font-size: .92rem; }
.footer { margin-top: 3rem; color: #7b8794; font-size: .85rem; }
@media print {
  body { margin: 0; max-width: none; }
  h2 { break-after: avoid; }
  .finding { break-inside: avoid; }
}
""".strip()


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _html_list(items: Sequence[Any]) -> str:
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str:
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_e(cell)}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _level_badge(level: int | None) -> str:
    if level is None:
        return '<span class="badge">Unscored</span>'
    return f'<span class="badge level-{level}">Level {level}</span>'


def render_html(data: ReportData) -> str:
    """Render the full report as a standalone HTML document."""
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(data.report_title)} — {_e(data.subject)}</title>",
        f"<style>{HTML_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{_e(data.report_title)} — {_e(data.subject)}</h1>",
        f'<p class="disclaimer">{_e(DISCLAIMER_TEXT)}</p>',
    ]

    document = data.document
    pdf_name = Path(str(document.get("pdf_path") or "")).name
    parts.append("<h2>Report details</h2>")
    parts.append(
        _html_table(
            ("Field", "Value"),
            [
                ("Document", pdf_name or "Not recorded"),
                (
                    "Pages parsed",
                    data.stats.get("pages", document.get("page_count", "—")),
                ),
                (
                    "Focus goals",
                    ", ".join(str(goal) for goal in data.focus_goals) or "—",
                ),
                ("Local judge", data.settings.get("model") or "Not recorded"),
                (
                    "Embedding model",
                    data.settings.get("embedding_model") or "Not recorded",
                ),
                ("Generated", data.generated_at),
                ("Tool", f"Prism17 {data.tool_version}".strip() or "Prism17"),
            ],
        )
    )

    stats = data.stats
    review_summary = summarise_reviews(data.scored, data.reviews)
    parts.append("<h2>Summary</h2>")
    parts.append(
        _html_table(
            ("Measure", "Value"),
            [
                ("Scored findings", stats.get("scored", len(data.scored))),
                ("Level distribution", _level_counts_text(stats)),
                (
                    "Targets with scored disclosure",
                    f"{data.coverage.targets_with_findings} of "
                    f"{data.coverage.targets_total} focus targets",
                ),
                ("Judged candidates", stats.get("judged", len(data.findings))),
                (
                    "Not relevant",
                    stats.get("not_relevant", len(data.not_relevant)),
                ),
                (
                    "Needs human review",
                    stats.get("needs_review", len(data.needs_review)),
                ),
                ("Unreadable pages", stats.get("unreadable_pages", 0)),
                ("Human review", _review_summary_text(review_summary)),
            ],
        )
    )
    parts.append(f"<p>{_e(COVERAGE_CAVEAT)}</p>")

    parts.append("<h2>Coverage by goal</h2>")
    if data.coverage.goals:
        parts.append(
            _html_table(
                (
                    "Goal",
                    "Focus area",
                    "Targets with scored disclosure",
                    "Scored findings",
                    "Best level",
                    "Status",
                ),
                [
                    (
                        f"Goal {goal.goal}",
                        goal.short_title,
                        f"{goal.covered_target_count} of {goal.targets_total}",
                        goal.finding_count,
                        _level_text(goal.best_level),
                        goal.status,
                    )
                    for goal in data.coverage.goals
                ],
            )
        )
    else:
        parts.append(
            "<p>No focus goals were available for coverage accounting.</p>"
        )

    parts.append("<h2>Evidence coverage matrix</h2>")
    parts.append(f"<p>{_e(EVIDENCE_MATRIX_NOTE)}</p>")
    if data.coverage.goals:
        parts.append(
            _html_table(
                _evidence_matrix_headers(),
                _evidence_matrix_rows(data.coverage),
            )
        )
    else:
        parts.append(
            "<p>No scored findings were available for dimension "
            "accounting.</p>"
        )

    parts.append("<h2>Scored findings</h2>")
    if not data.scored:
        parts.append(
            "<p>No scored findings. No finding met both quote verification "
            "and rubric requirements.</p>"
        )
    else:
        goals, targets = _coverage_indexes(data.coverage)
        grouped = _group_by_goal(data.scored)
        for goal_number in sorted(grouped):
            goal_cov = goals.get(goal_number)
            if goal_cov is not None:
                parts.append(
                    f"<h3>Goal {goal_number} — {_e(goal_cov.short_title)}</h3>"
                )
                if goal_cov.goal_title:
                    parts.append(f"<p>{_e(goal_cov.goal_title)}</p>")
                parts.append(
                    f"<p><strong>Goal status:</strong> {_e(goal_cov.status)} "
                    f"({_plural(goal_cov.finding_count, 'scored finding')}; "
                    f"{goal_cov.covered_target_count} of "
                    f"{goal_cov.targets_total} targets with scored "
                    "disclosure).</p>"
                )
                for text in guidance_for_goal(goal_cov):
                    parts.append(
                        f"<p><strong>Goal guidance:</strong> {_e(text)}</p>"
                    )
            else:
                parts.append(f"<h3>Goal {goal_number}</h3>")

            for target_id in sorted(grouped[goal_number], key=target_sort_key):
                target_cov = targets.get((goal_number, target_id))
                target_findings = grouped[goal_number][target_id]
                best_level = max(
                    (
                        finding.level
                        for finding in target_findings
                        if finding.level is not None
                    ),
                    default=None,
                )
                parts.append(
                    f"<h4>Target {_e(target_id)} "
                    f"{_level_badge(best_level)}</h4>"
                )
                if target_cov is not None and target_cov.target_text:
                    parts.append(
                        f'<p class="target-text">'
                        f"<em>Target:</em> {_e(_squash(target_cov.target_text))}"
                        "</p>"
                    )
                for index, finding in enumerate(target_findings, start=1):
                    label = (
                        f"Finding {index} — page {finding.source_page}"
                        if len(target_findings) > 1
                        else f"Page {finding.source_page}"
                    )
                    method = finding.quote_method or "unknown method"
                    parts.append('<div class="finding">')
                    parts.append(
                        f'<p class="meta">{_e(label)} · quote verified '
                        f"({_e(method)} {finding.quote_score:.1f})</p>"
                    )
                    parts.append(
                        f"<blockquote>{_e(_squash(finding.judged.supporting_quote))}"
                        "</blockquote>"
                    )
                    parts.append(
                        "<p><strong>Assessment:</strong> "
                        f"{_e(_squash(finding.level_rationale))}</p>"
                    )
                    if finding.disclosure_gaps:
                        parts.append("<p><strong>Disclosure gaps:</strong></p>")
                        parts.append(_html_list(finding.disclosure_gaps))
                    else:
                        parts.append(
                            "<p><strong>Disclosure gaps:</strong> none "
                            "identified by the rubric.</p>"
                        )
                    suggestions = guidance_for_finding(finding)
                    if suggestions:
                        parts.append(
                            "<p><strong>Suggested improvements:</strong></p>"
                        )
                        parts.append(_html_list(suggestions))
                    else:
                        parts.append(
                            "<p><strong>Suggested improvements:</strong> none "
                            "identified by the rubric.</p>"
                        )
                    review_label = _review_label(
                        data.reviews.get(finding_key(finding))
                    )
                    if review_label:
                        parts.append(
                            "<p><strong>Human review:</strong> "
                            f"{_e(review_label)}</p>"
                        )
                    parts.append("</div>")

    review_rows = [
        row
        for row in export_rows(data.scored, data.reviews)
        if row["review_status"] != "pending"
    ]
    if review_rows:
        parts.append("<h2>Human review</h2>")
        parts.append(
            _html_table(
                (
                    "Target",
                    "Page",
                    "Rubric level",
                    "Status",
                    "Reviewed level",
                    "Reviewer",
                    "Note",
                ),
                [
                    (
                        row["target"],
                        row["page"],
                        _level_text(row["rubric_level"]),
                        row["review_status"],
                        (
                            f"Level {row['review_level']}"
                            if row["review_level"]
                            else "—"
                        ),
                        row["reviewer"],
                        row["note"],
                    )
                    for row in review_rows
                ],
            )
        )

    goals_without = [
        goal for goal in data.coverage.goals if goal.finding_count == 0
    ]
    if goals_without:
        parts.append("<h2>Goals without scored findings</h2>")
        for goal in goals_without:
            parts.append(
                f"<h3>Goal {goal.goal} — {_e(goal.short_title)}</h3>"
            )
            if goal.goal_title:
                parts.append(f"<p>{_e(goal.goal_title)}</p>")
            parts.append(f"<p>{_e(NO_DISCLOSURE_GUIDANCE)}</p>")

    if data.include_unscored and data.needs_review:
        parts.append("<h2>Needs human review</h2>")
        parts.append(
            _html_list(
                [
                    f"Page {finding.source_page}, target "
                    f"{finding.judged.sdg_target}: "
                    f"{finding.review_reason or 'not scored'}"
                    for finding in data.needs_review
                ]
            )
        )

    if data.not_relevant:
        parts.append("<h2>Candidates excluded as not relevant</h2>")
        count = len(data.not_relevant)
        verb = "was" if count == 1 else "were"
        parts.append(
            f"<p>{_plural(count, 'candidate')} {verb} judged not relevant to "
            f"the retrieved target and {verb} not scored. This count describes "
            "retrieval and judging output, not the company.</p>"
        )

    parts.append("<h2>Method and limitations</h2>")
    parts.append(_html_list([f"{label}: {text}" for label, text in METHOD_ITEMS]))
    unreadable = int(stats.get("unreadable_pages", 0) or 0)
    if unreadable:
        parts.append(
            f"<p><strong>Unreadable pages:</strong> {unreadable} page(s) had no "
            "usable text layer and were not assessed.</p>"
        )

    parts.append("<h2>Responsible use</h2>")
    parts.append(
        _html_list(
            [
                "A disclosure gap is not proof of poor performance.",
                "A high level reflects measurable, time-bound, "
                "baseline-anchored disclosure, not the quality of the "
                "underlying activity.",
                "Quotes are checked against the source page; semantic support "
                "and rubric decisions should be reviewed by a human.",
                "This report must not be used to make automated accusations "
                "about a company.",
            ]
        )
    )
    parts.append(
        f'<p class="footer">Generated {_e(data.generated_at)} by Prism17 '
        f"{_e(data.tool_version)}. Local, open-source pipeline.</p>"
    )
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def _write_text(
    text: str,
    path: str | Path,
    config: dict[str, Any] | None,
) -> Path:
    target = resolve_path(path, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def write_markdown_report(
    data: ReportData,
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    return _write_text(render_markdown(data), path, config)


def write_html_report(
    data: ReportData,
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    return _write_text(render_html(data), path, config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Phase 3 SDG Alignment Report (Markdown and/or HTML) "
            "from a Phase 2 audit JSON file."
        )
    )
    parser.add_argument("audit", help="Path to the Phase 2 audit JSON")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write Markdown report here")
    parser.add_argument(
        "--html", default=None, help="Write standalone HTML report here"
    )
    parser.add_argument("--company", default=None, help="Override company name")
    parser.add_argument("--title", default=None, help="Override report title")
    parser.add_argument(
        "--reviews",
        default=None,
        help="Path to a review decisions JSON to overlay on the report",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        audit_path = resolve_path(args.audit, config)
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        reviews = (
            load_reviews(resolve_path(args.reviews, config))
            if args.reviews
            else {}
        )
        data = report_data_from_json(payload, config, reviews=reviews)
        if args.company:
            data.company_name = args.company
        if args.title:
            data.report_title = args.title

        wrote = False
        if args.out:
            path = write_markdown_report(data, args.out, config)
            print(f"Wrote Markdown report: {path}")
            wrote = True
        if args.html:
            path = write_html_report(data, args.html, config)
            print(f"Wrote HTML report: {path}")
            wrote = True
        if not wrote:
            print(render_markdown(data))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
