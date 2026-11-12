"""Narrow multi-year tracking for quantified SDG targets.

Only scored findings at Level 3 or 4 are compared by default; ``min_level`` can
change that threshold. The module never concludes that a target was discontinued:
a target absent from a later report is labelled ``not_found_in_later_report`` and
needs human review. Units, baselines, and values must be comparable before any
direction is reported.

Statuses are neutral:

* ``progressing`` - the value moved in the direction the target asks for.
* ``unchanged`` - the value is the same within a small tolerance.
* ``changed`` - the value moved, but not in a direction that can be called
  progress (or the direction is unknown).
* ``not_comparable`` - units, baselines, or values cannot be aligned.
* ``not_found_in_later_report`` - no matching quantified target in the later
  report. This is not evidence of discontinuation.
* ``new_in_later_report`` - the later report has a quantified target that the
  earlier one did not.

Run:
    python -m prism17.tracking outputs/earlier.audit.json outputs/later.audit.json \
        --out outputs/tracking.json --markdown outputs/tracking.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import coerce_rubric_level, get, load_config, resolve_path
from .models import VerifiedFinding

LOGGER = logging.getLogger(__name__)

TRACKING_STATUS_TEXT = {
    "progressing": "progressing",
    "unchanged": "unchanged",
    "changed": "changed",
    "not_comparable": "not comparable",
    "not_found_in_later_report": "not found in later report (needs review)",
    "new_in_later_report": "new in later report",
}

LEVELS = (1, 2, 3, 4)

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_FY_RANGE_RE = re.compile(
    r"(?<![A-Za-z])FY\s*((?:19|20)\d{2})\s*[-_/]\s*(\d{2,4})(?!\d)",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "percent": "%",
    "per cent": "%",
    "percentage": "%",
    "tco2": "tco2e",
    "mtco2e": "tco2e",
    "tonnes": "t",
    "tonne": "t",
    "tons": "t",
    "ton": "t",
    "litres": "l",
    "litre": "l",
    "liters": "l",
    "liter": "l",
}

_MULTIPLIER_ALIASES = {
    "thousand": "thousand",
    "thousands": "thousand",
    "million": "million",
    "millions": "million",
    "billion": "billion",
    "billions": "billion",
    "lakh": "lakh",
    "lakhs": "lakh",
    "crore": "crore",
    "crores": "crore",
}

_UNIT_TOKENS = (
    "mtco2e",
    "ktco2e",
    "tco2e",
    "mwh",
    "gwh",
    "kwh",
    "gj",
    "kl",
    "m3",
    "tonnes",
    "tonne",
    "tons",
    "ton",
    "kg",
    "inr",
    "t",
    "l",
)

_DECREASE_CUES = re.compile(
    r"\b(?:reduce|reduction|reduced|lower|lowering|cut|cutting|decrease|"
    r"decreasing|decline|halve|halved|minimi[sz]e|less|emissions?|"
    r"footprint|intensity|waste|water use|energy use|loss)\b",
    re.IGNORECASE,
)
_INCREASE_CUES = re.compile(
    r"\b(?:increase|increasing|raise|raising|grow|growing|expand|expanding|"
    r"double|triple|install|capacity|coverage|share|renewable|recycled|"
    r"recycling|plantation|trees?|training|participation|access)\b",
    re.IGNORECASE,
)


@dataclass
class QuantifiedValue:
    """One quantified observation from one audit report."""

    value: float | None
    raw_value: str
    unit: str
    year: str
    status: str
    page: int
    quote: str
    claim: str
    level: int
    target_text: str
    passage_id: str
    baseline: str = ""

    def display(self) -> str:
        if self.value is None:
            return self.raw_value or "value not extracted"
        number = f"{self.value:g}"
        return f"{number} {self.unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw_value": self.raw_value,
            "unit": self.unit,
            "year": self.year,
            "status": self.status,
            "page": self.page,
            "quote": self.quote,
            "claim": self.claim,
            "level": self.level,
            "target_text": self.target_text,
            "passage_id": self.passage_id,
            "baseline": self.baseline,
        }


@dataclass
class TrackedTarget:
    """Comparison result for one SDG target across two reports."""

    target_id: str
    goal: int
    indicator: str
    earlier: QuantifiedValue | None = None
    later: QuantifiedValue | None = None
    status: str = "not_comparable"
    reason: str = ""
    direction: str = "unknown"
    change: float | None = None
    change_percent: float | None = None

    @property
    def status_text(self) -> str:
        return TRACKING_STATUS_TEXT.get(self.status, self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "goal": self.goal,
            "indicator": self.indicator,
            "earlier": self.earlier.to_dict() if self.earlier else None,
            "later": self.later.to_dict() if self.later else None,
            "status": self.status,
            "status_text": self.status_text,
            "reason": self.reason,
            "direction": self.direction,
            "change": self.change,
            "change_percent": self.change_percent,
        }


@dataclass
class TrackingReport:
    """All target comparisons for one pair of reports."""

    records: list[TrackedTarget] = field(default_factory=list)
    earlier_label: str = "earlier report"
    later_label: str = "later report"
    min_level: int = 3
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in TRACKING_STATUS_TEXT}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "earlier_label": self.earlier_label,
            "later_label": self.later_label,
            "min_level": self.min_level,
            "counts": self.counts(),
            "records": [record.to_dict() for record in self.records],
            "notes": list(self.notes),
        }


def normalise_unit(unit: str) -> str:
    value = " ".join(str(unit or "").strip().lower().split())
    if value in _UNIT_ALIASES:
        return _UNIT_ALIASES[value]
    parts = value.split()
    if len(parts) == 2 and parts[0] in _MULTIPLIER_ALIASES:
        base = _UNIT_ALIASES.get(parts[1], parts[1])
        return f"{_MULTIPLIER_ALIASES[parts[0]]} {base}"
    return value


def _first_number(text: str, *, ignore_years: bool = False) -> float | None:
    source = _without_years(text) if ignore_years else str(text or "")
    match = _NUMBER_RE.search(source)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _last_number(text: str, *, max_gap: int) -> float | None:
    """Return the last number in ``text`` if it ends near the text boundary."""
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    if len(text) - last.end() > max_gap:
        return None
    try:
        return float(last.group(0).replace(",", ""))
    except ValueError:
        return None


def _number_near_unit(text: str, *, max_gap: int = 24) -> float | None:
    """Return the number written immediately before a recognised unit.

    Used only when the judge did not extract a value elsewhere. Binding the
    number to the unit avoids treating an unrelated count (for example "1
    million trees") as the value for a later unit such as tonnes.
    """
    cleaned = _without_years(text)
    lowered = cleaned.lower()
    best: tuple[int, float] | None = None
    for token in _UNIT_TOKENS + ("%",):
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
            lowered,
        ):
            number = _last_number(cleaned[: match.start()], max_gap=max_gap)
            if number is None:
                continue
            if best is None or match.start() < best[0]:
                best = (match.start(), number)
    return best[1] if best else None


def _without_years(text: str) -> str:
    """Remove year and financial-year tokens before value fallback."""
    without_ranges = _FY_RANGE_RE.sub(" ", str(text or ""))
    return _YEAR_RE.sub(" ", without_ranges)


def _has_unit_token(lowered: str, token: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z]){re.escape(token)}(?![a-z0-9])", lowered
        )
        is not None
    )


def _unit_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    if (
        "%" in lowered
        or _has_unit_token(lowered, "percent")
        or _has_unit_token(lowered, "percentage")
        or "per cent" in lowered
    ):
        return "%"

    multiplier_pattern = "|".join(_MULTIPLIER_ALIASES)
    for token in _UNIT_TOKENS:
        compound = re.search(
            rf"(?<![a-z])({multiplier_pattern})\s+{re.escape(token)}"
            rf"(?![a-z0-9])",
            lowered,
        )
        if compound:
            multiplier = _MULTIPLIER_ALIASES[compound.group(1)]
            return f"{multiplier} {normalise_unit(token)}"

    for token in _UNIT_TOKENS:
        if _has_unit_token(lowered, token):
            return normalise_unit(token)
    return ""


def infer_reporting_year(document: dict[str, Any] | None) -> int | None:
    """Infer a reporting year from a document title or file name.

    ``FY2023-24`` is mapped to 2024 (the year the financial year ends), which is
    the convention used in the sample BRSR. Callers can always override it.
    """
    if not document:
        return None
    text = " ".join(
        str(document.get(key) or "")
        for key in ("title", "pdf_path", "file_name", "name")
    )
    fy_match = _FY_RANGE_RE.search(text)
    if fy_match:
        start = int(fy_match.group(1))
        end_text = fy_match.group(2)
        if len(end_text) == 2:
            end = (start // 100) * 100 + int(end_text)
            if end < start:
                end += 100
        else:
            end = int(end_text)
        return end
    years = [int(match.group(1)) for match in _YEAR_RE.finditer(text)]
    return max(years) if years else None


def _value_from_finding(
    finding: VerifiedFinding,
    *,
    year: str = "",
) -> QuantifiedValue:
    extracted = finding.judged.extracted or {}
    raw_value = str(extracted.get("value", "") or "").strip()
    unit = normalise_unit(str(extracted.get("unit", "") or ""))
    value = _first_number(raw_value)
    quote = str(finding.judged.supporting_quote or "")
    if value is None:
        # Fall back to the first number in the quote only when the judge did
        # not record a value, the quote contains a recognisable unit, and year
        # tokens are ignored. This avoids treating a reporting year as a value.
        quote_unit = _unit_from_text(quote)
        if quote_unit:
            value = _number_near_unit(quote)
            if value is not None:
                raw_value = f"{value:g}"
    if not unit:
        unit = _unit_from_text(raw_value) or _unit_from_text(quote)

    return QuantifiedValue(
        value=value,
        raw_value=raw_value,
        unit=unit,
        year=str(extracted.get("year", "") or year or ""),
        status=str(extracted.get("status", "unclear") or "unclear"),
        page=finding.source_page,
        quote=quote,
        claim=str(finding.judged.claim or ""),
        level=int(finding.level or 0),
        target_text=str(finding.judged.candidate.target_text or ""),
        passage_id=finding.judged.candidate.passage_id,
        baseline=str(extracted.get("baseline", "") or "").strip(),
    )


def _representatives(
    findings: Sequence[VerifiedFinding],
    *,
    min_level: int,
    year: str = "",
) -> dict[str, QuantifiedValue]:
    """Choose one quantified finding per target.

    When a target has several quantified findings, the highest rubric level
    wins; ties go to the higher judge confidence, then the earlier passage id.
    This is deliberately deterministic and documented, not a silent average.
    """
    candidates: dict[str, list[tuple[tuple[Any, ...], QuantifiedValue]]] = {}
    for finding in findings:
        if not finding.is_scored:
            continue
        if finding.level is None or int(finding.level) < min_level:
            continue
        target_id = str(finding.judged.sdg_target or "").strip()
        if not target_id:
            continue
        value = _value_from_finding(finding, year=year)
        rank = (
            -int(finding.level),
            -float(finding.judged.confidence),
            value.page,
            value.passage_id,
        )
        candidates.setdefault(target_id, []).append((rank, value))

    chosen: dict[str, QuantifiedValue] = {}
    for target_id, items in candidates.items():
        items.sort(key=lambda item: item[0])
        chosen[target_id] = items[0][1]
    return chosen


def infer_direction(earlier: QuantifiedValue) -> str:
    """Infer whether lower or higher values are favourable for a target."""
    text = " ".join(
        [earlier.target_text, earlier.claim, earlier.quote]
    ).strip()
    decrease = bool(_DECREASE_CUES.search(text))
    increase = bool(_INCREASE_CUES.search(text))
    if decrease and not increase:
        return "decrease"
    if increase and not decrease:
        return "increase"
    return "unknown"


def _is_close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-9)


def _compare_values(
    target_id: str,
    goal: int,
    earlier: QuantifiedValue,
    later: QuantifiedValue,
    config: dict[str, Any],
) -> TrackedTarget:
    tracking_cfg = get(config, "tracking", default={}) or {}
    require_same_unit = bool(tracking_cfg.get("require_same_unit", True))

    record = TrackedTarget(
        target_id=target_id,
        goal=goal,
        indicator=earlier.claim or earlier.target_text,
        earlier=earlier,
        later=later,
        direction=infer_direction(earlier),
    )

    if earlier.value is None or later.value is None:
        record.status = "not_comparable"
        record.reason = "a numeric value was not extracted from one report"
        return record

    if require_same_unit and earlier.unit != later.unit:
        if not earlier.unit or not later.unit:
            record.status = "not_comparable"
            record.reason = (
                "a unit was not extracted from one report; values were not "
                "compared"
            )
            return record
        record.status = "not_comparable"
        record.reason = (
            f"unit changed from {earlier.unit} to {later.unit}; "
            "values were not compared"
        )
        return record

    if earlier.baseline and later.baseline and earlier.baseline != later.baseline:
        record.status = "not_comparable"
        record.reason = (
            f"baseline changed from {earlier.baseline} to {later.baseline}; "
            "values were not compared"
        )
        return record

    change = later.value - earlier.value
    record.change = round(change, 6)
    if earlier.value != 0:
        record.change_percent = round(change / earlier.value * 100, 3)

    if _is_close(earlier.value, later.value):
        record.status = "unchanged"
        record.reason = "the reported value is the same within tolerance"
        return record

    if record.direction == "decrease":
        favourable = change < 0
    elif record.direction == "increase":
        favourable = change > 0
    else:
        record.status = "changed"
        record.reason = (
            "the reported value changed, but the favourable direction could "
            "not be inferred from the target wording"
        )
        return record

    if favourable:
        record.status = "progressing"
        record.reason = (
            f"the reported value moved in the favourable direction for a "
            f"{record.direction} indicator"
        )
    else:
        record.status = "changed"
        record.reason = (
            f"the reported value moved against the favourable direction for a "
            f"{record.direction} indicator"
        )
    return record


def build_tracking(
    earlier_findings: Sequence[VerifiedFinding],
    later_findings: Sequence[VerifiedFinding],
    config: dict[str, Any],
    *,
    earlier_label: str = "earlier report",
    later_label: str = "later report",
    earlier_year: int | None = None,
    later_year: int | None = None,
    min_level: int | None = None,
) -> TrackingReport:
    """Compare quantified targets between two completed audits."""
    tracking_cfg = get(config, "tracking", default={}) or {}
    level = int(
        min_level
        if min_level is not None
        else tracking_cfg.get("min_level", 3)
    )
    if level not in LEVELS:
        raise ValueError(
            f"tracking min_level must be one of {LEVELS}, got {level}"
        )
    early = _representatives(
        earlier_findings, min_level=level, year=str(earlier_year or "")
    )
    late = _representatives(
        later_findings, min_level=level, year=str(later_year or "")
    )

    records: list[TrackedTarget] = []
    for target_id in sorted(early, key=_target_sort_key):
        earlier = early[target_id]
        later = late.get(target_id)
        if later is None:
            records.append(
                TrackedTarget(
                    target_id=target_id,
                    goal=_goal_from_target(target_id),
                    indicator=earlier.claim or earlier.target_text,
                    earlier=earlier,
                    status="not_found_in_later_report",
                    reason=(
                        "no matching quantified finding in the later report; "
                        "this is not evidence that the target was discontinued"
                    ),
                    direction=infer_direction(earlier),
                )
            )
            continue
        records.append(
            _compare_values(
                target_id,
                _goal_from_target(target_id),
                earlier,
                later,
                config,
            )
        )

    for target_id in sorted(set(late) - set(early), key=_target_sort_key):
        later = late[target_id]
        records.append(
            TrackedTarget(
                target_id=target_id,
                goal=_goal_from_target(target_id),
                indicator=later.claim or later.target_text,
                later=later,
                status="new_in_later_report",
                reason="no comparable quantified finding in the earlier report",
                direction=infer_direction(later),
            )
        )

    report = TrackingReport(
        records=records,
        earlier_label=earlier_label,
        later_label=later_label,
        min_level=level,
    )
    report.notes = [
        f"Only Level {level} and above scored findings are compared.",
        "A target that is absent from the later report is marked "
        "'not found in later report (needs review)', never discontinued.",
        "A changed value is labelled 'changed' unless the target wording "
        "supports a clear favourable direction.",
        "Values are compared only when both reports have a recognised, "
        "matching unit; when both reports record a baseline, those baselines "
        "must also agree.",
        "Review the page numbers and quotes in both audits before using a "
        "comparison.",
    ]
    return report


def _goal_from_target(target_id: str) -> int:
    try:
        return int(str(target_id).split(".", 1)[0])
    except (ValueError, IndexError):
        return 0


def _target_sort_key(target_id: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"([0-9]+)", str(target_id))
        if part
    )


def _findings_from_payload(payload: dict[str, Any]) -> list[VerifiedFinding]:
    if "findings" not in payload:
        raise ValueError("audit JSON has no 'findings' list")
    return [
        VerifiedFinding.from_dict(item)
        for item in (payload.get("findings") or [])
    ]


def track_payloads(
    earlier_payload: dict[str, Any],
    later_payload: dict[str, Any],
    config: dict[str, Any],
    *,
    earlier_label: str | None = None,
    later_label: str | None = None,
    earlier_year: int | None = None,
    later_year: int | None = None,
    min_level: int | None = None,
) -> TrackingReport:
    """Track two audit JSON payloads."""
    early_document = earlier_payload.get("document") or {}
    late_document = later_payload.get("document") or {}
    early_label = earlier_label or _label(early_document, "earlier report")
    late_label = later_label or _label(late_document, "later report")
    return build_tracking(
        _findings_from_payload(earlier_payload),
        _findings_from_payload(later_payload),
        config,
        earlier_label=early_label,
        later_label=late_label,
        earlier_year=(
            earlier_year
            if earlier_year is not None
            else infer_reporting_year(early_document)
        ),
        later_year=(
            later_year
            if later_year is not None
            else infer_reporting_year(late_document)
        ),
        min_level=min_level,
    )


def _label(document: dict[str, Any], fallback: str) -> str:
    title = str(document.get("title") or "").strip()
    if title:
        return title
    pdf_path = str(document.get("pdf_path") or "").strip()
    return Path(pdf_path).stem if pdf_path else fallback


def render_markdown(report: TrackingReport) -> str:
    """Render a compact, neutral tracking report."""
    lines = [
        f"# Multi-year target tracking: {report.earlier_label} -> {report.later_label}",
        "",
        "> This compares **quantified disclosure** between two audits. It does "
        "not verify real-world performance, and a missing target is not "
        "evidence that the target was discontinued.",
        "",
        "## Summary",
        "",
    ]
    counts = report.counts()
    for status, label in TRACKING_STATUS_TEXT.items():
        lines.append(f"- {label}: {counts.get(status, 0)}")
    lines.append("")

    if not report.records:
        lines.append(
            f"No quantified findings at Level {report.min_level} or above "
            "were available."
        )
        lines.append("")
    else:
        lines.extend(
            [
                "| Target | Indicator | Earlier | Later | Change | Status | Reason |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for record in report.records:
            earlier = record.earlier.display() if record.earlier else "n/a"
            later = record.later.display() if record.later else "n/a"
            change = (
                f"{record.change:g}"
                + (
                    f" ({record.change_percent:+g}%)"
                    if record.change_percent is not None
                    else ""
                )
                if record.change is not None
                else "n/a"
            )
            indicator = " ".join(record.indicator.split())
            reason = " ".join(record.reason.split())
            lines.append(
                f"| {record.target_id} | {indicator} | {earlier} | {later} | "
                f"{change} | {record.status_text} | {reason} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(report: TrackingReport, path: str | Path, config: dict[str, Any]) -> Path:
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
            "Compare quantified SDG targets across two Prism17 audit JSON files."
        )
    )
    parser.add_argument("earlier", help="Earlier audit JSON")
    parser.add_argument("later", help="Later audit JSON")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write tracking JSON here")
    parser.add_argument(
        "--markdown", default=None, help="Write a Markdown tracking report here"
    )
    parser.add_argument("--earlier-label", default=None)
    parser.add_argument("--later-label", default=None)
    parser.add_argument("--earlier-year", type=int, default=None)
    parser.add_argument("--later-year", type=int, default=None)
    parser.add_argument(
        "--min-level",
        type=int,
        default=None,
        help="Minimum rubric level to compare (default: config tracking.min_level)",
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
        earlier_path = resolve_path(args.earlier, config)
        later_path = resolve_path(args.later, config)
        earlier_payload = json.loads(earlier_path.read_text(encoding="utf-8"))
        later_payload = json.loads(later_path.read_text(encoding="utf-8"))
        report = track_payloads(
            earlier_payload,
            later_payload,
            config,
            earlier_label=args.earlier_label,
            later_label=args.later_label,
            earlier_year=args.earlier_year,
            later_year=args.later_year,
            min_level=args.min_level,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(render_markdown(report))
    if args.out:
        path = _write_json(report, args.out, config)
        print(f"\nWrote tracking JSON: {path}")
    if args.markdown:
        path = resolve_path(args.markdown, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote tracking Markdown: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
