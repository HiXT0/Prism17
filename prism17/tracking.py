"""Narrow multi-year tracking for quantified SDG targets.

Only scored findings at Level 3 or 4 are compared by default; ``min_level`` can
change that threshold. The module never concludes that a target was
discontinued: a target absent from a later report is labelled
``not_found_in_later_report`` and needs human review. Units, baselines, and
values must be comparable before any direction is reported.

Statuses are neutral:

* ``progressing`` - the value moved in the direction the target asks for.
* ``unchanged`` - the two values are equal within a small tolerance.
* ``changed`` - the value changed, but the favourable direction is unclear.
* ``not_comparable`` - units, baselines, or values cannot be compared.
* ``not_found_in_later_report`` - present earlier, absent later; needs review.
* ``new_in_later_report`` - present only in the later report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from .audit_io import findings_from_payload, load_payload
from .config import coerce_rubric_level, get, load_config, resolve_path
from .coverage import target_sort_key
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
    r"(?<![A-Za-z])FY\s*((?:19|20)\d{2})\s*[-_/]\s*(\d{2,4})(?!\d)"
)

_UNIT_ALIASES = {
    "tco2": "tco2e",
    "tco2e": "tco2e",
    "co2e": "tco2e",
    "mtco2e": "mtco2e",
    "ktco2e": "ktco2e",
    "tonnes": "t",
    "tonne": "t",
    "tons": "t",
    "ton": "t",
    "t": "t",
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "per cent": "%",
    "mwh": "mwh",
    "gwh": "gwh",
    "kwh": "kwh",
    "gj": "gj",
    "kl": "kl",
    "m3": "m3",
    "kg": "kg",
    "liters": "l",
    "liter": "l",
    "litres": "l",
    "litre": "l",
    "l": "l",
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
    r"footprint|intensity|waste|water use|energy use|loss)\b"
)

_INCREASE_CUES = re.compile(
    r"\b(?:increase|increasing|expand|expanding|improve|improving|double|"
    r"grow|growing|achieve|ensure|access|share|coverage|participation|"
    r"investment|spending|training|renewable)\b"
)


@dataclass
class QuantifiedValue:
    """One representative quantified finding for a target."""

    target_id: str
    target_goal: int
    target_text: str
    level: int
    claim: str
    quote: str
    page: int
    passage_id: str
    value: float | None
    raw_value: str
    unit: str
    year: str
    baseline: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_goal": self.target_goal,
            "target_text": self.target_text,
            "level": self.level,
            "claim": self.claim,
            "quote": self.quote,
            "page": self.page,
            "passage_id": self.passage_id,
            "value": self.value,
            "raw_value": self.raw_value,
            "unit": self.unit,
            "year": self.year,
            "baseline": self.baseline,
            "confidence": self.confidence,
        }


@dataclass
class TrackingRecord:
    """Comparison result for one SDG target."""

    target_id: str
    target_goal: int
    target_text: str
    status: str
    earlier: QuantifiedValue | None = None
    later: QuantifiedValue | None = None
    change: float | None = None
    change_pct: float | None = None
    direction: str | None = None
    reason: str = ""

    @property
    def status_text(self) -> str:
        return TRACKING_STATUS_TEXT.get(self.status, self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_goal": self.target_goal,
            "target_text": self.target_text,
            "status": self.status,
            "status_text": self.status_text,
            "earlier": self.earlier.to_dict() if self.earlier else None,
            "later": self.later.to_dict() if self.later else None,
            "change": self.change,
            "change_pct": self.change_pct,
            "direction": self.direction,
            "reason": self.reason,
        }


@dataclass
class TrackingReport:
    """All tracking records and the comparison context."""

    records: list[TrackingRecord] = field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Value and unit extraction
# ---------------------------------------------------------------------------


def normalise_unit(unit: str) -> str:
    """Normalise a unit label without converting magnitudes."""
    value = " ".join(str(unit or "").strip().lower().split())
    if value in _UNIT_ALIASES:
        return _UNIT_ALIASES[value]
    parts = value.split()
    if len(parts) == 2 and parts[0] in _MULTIPLIER_ALIASES:
        base = _UNIT_ALIASES.get(parts[1], parts[1])
        return f"{_MULTIPLIER_ALIASES[parts[0]]} {base}"
    return value


def _has_unit_token(lowered: str, token: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z]){re.escape(token)}(?![a-z0-9])", lowered
        )
        is not None
    )


def _without_years(text: str) -> str:
    """Remove year and financial-year tokens before value fallback."""
    without_ranges = _FY_RANGE_RE.sub(" ", str(text or ""))
    return _YEAR_RE.sub(" ", without_ranges)


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
    number to the unit avoids treating an unrelated count as the value for a
    later unit such as tonnes.
    """
    cleaned = _without_years(text)
    lowered = cleaned.lower()
    best: tuple[int, float] | None = None
    for token in (*_UNIT_TOKENS, "%"):
        for match in re.finditer(
            rf"(?<![a-z]){re.escape(token)}(?![a-z0-9])",
            lowered,
        ):
            number = _last_number(cleaned[: match.start()], max_gap=max_gap)
            if number is None:
                continue
            if best is None or match.start() < best[0]:
                best = (match.start(), number)
    return best[1] if best else None


def _unit_from_text(text: str) -> str:
    """Return a recognised unit from free text, preserving multipliers."""
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


def _value_from_finding(
    finding: VerifiedFinding,
    *,
    year: str = "",
) -> QuantifiedValue:
    extracted = finding.judged.extracted or {}
    raw_value = str(extracted.get("value", "") or "").strip()
    unit = normalise_unit(str(extracted.get("unit", "") or "").strip())
    quote = str(finding.judged.supporting_quote or "")
    value = _first_number(raw_value)

    if value is not None and not unit:
        # A year token in the value field is not a measurement. Keep it only
        # when an explicit unit makes the number a quantity.
        if _first_number(raw_value, ignore_years=True) is None:
            value = None

    if value is None:
        # Fall back to a number bound to a unit in the quote. If the quote has
        # no unit, use the first non-year number as a last resort.
        quote_unit = _unit_from_text(quote)
        if quote_unit:
            value = _number_near_unit(quote)
        if value is None:
            value = _first_number(quote, ignore_years=True)
        if value is not None:
            raw_value = f"{value:g}"

    if not unit:
        unit = _unit_from_text(raw_value) or _unit_from_text(quote)

    baseline = str(extracted.get("baseline", "") or "").strip() or None
    return QuantifiedValue(
        target_id=str(finding.judged.sdg_target or "").strip(),
        target_goal=int(finding.judged.sdg_goal or 0),
        target_text=str(finding.judged.candidate.target_text or ""),
        level=int(finding.level or 0),
        claim=str(finding.judged.claim or ""),
        quote=quote,
        page=int(finding.source_page or 0),
        passage_id=str(finding.judged.candidate.passage_id or ""),
        value=value,
        raw_value=raw_value,
        unit=unit,
        year=str(extracted.get("year", "") or year or ""),
        baseline=baseline,
        confidence=float(finding.judged.confidence or 0.0),
    )


# ---------------------------------------------------------------------------
# Representative selection and comparison
# ---------------------------------------------------------------------------


def _representatives(
    findings: Sequence[VerifiedFinding],
    *,
    min_level: int,
) -> dict[str, QuantifiedValue]:
    candidates: dict[str, list[tuple[tuple[float, float, int, str], QuantifiedValue]]] = {}
    for finding in findings:
        if not finding.is_scored:
            continue
        if finding.level is None or int(finding.level) < min_level:
            continue
        target_id = str(finding.judged.sdg_target or "").strip()
        if not target_id:
            continue
        value = _value_from_finding(finding)
        rank = (
            -int(finding.level),
            -float(finding.judged.confidence or 0.0),
            int(finding.source_page or 0),
            str(finding.judged.candidate.passage_id or ""),
        )
        candidates.setdefault(target_id, []).append((rank, value))

    chosen: dict[str, QuantifiedValue] = {}
    for target_id, items in candidates.items():
        items.sort(key=lambda item: item[0])
        chosen[target_id] = items[0][1]
    return chosen


def _direction_from_target(target_text: str) -> str | None:
    lowered = str(target_text or "").lower()
    decrease = bool(_DECREASE_CUES.search(lowered))
    increase = bool(_INCREASE_CUES.search(lowered))
    if decrease and not increase:
        return "decrease"
    if increase and not decrease:
        return "increase"
    return None


def _format_value(item: QuantifiedValue) -> str:
    if item.value is None:
        return "no numeric value"
    unit = f" {item.unit}" if item.unit else ""
    return f"{item.value:g}{unit}"


def _compare(
    earlier: QuantifiedValue,
    later: QuantifiedValue,
    *,
    target_text: str,
    require_same_unit: bool,
) -> TrackingRecord:
    record = TrackingRecord(
        target_id=earlier.target_id,
        target_goal=earlier.target_goal or later.target_goal,
        target_text=target_text or earlier.target_text or later.target_text,
        status="not_comparable",
        earlier=earlier,
        later=later,
    )

    if earlier.value is None or later.value is None:
        record.reason = "missing numeric value in one report"
        return record

    if require_same_unit:
        if not earlier.unit or not later.unit:
            record.reason = "missing recognized unit in one report"
            return record
        if earlier.unit != later.unit:
            record.reason = (
                f"unit changed ({earlier.unit} to {later.unit})"
            )
            return record

    if earlier.baseline and later.baseline:
        if earlier.baseline.strip().lower() != later.baseline.strip().lower():
            record.reason = (
                f"baseline changed ({earlier.baseline} to {later.baseline})"
            )
            return record

    change = float(later.value) - float(earlier.value)
    record.change = round(change, 6)
    if earlier.value != 0:
        record.change_pct = round((change / earlier.value) * 100, 4)

    if abs(change) <= 1e-9:
        record.status = "unchanged"
        record.direction = None
        record.reason = "values are equal"
        return record

    direction = "increase" if change > 0 else "decrease"
    record.direction = direction
    favourable = _direction_from_target(target_text)
    if favourable is None:
        record.status = "changed"
        record.reason = (
            "value changed, but the target wording does not identify a "
            "clear favourable direction"
        )
    elif direction == favourable:
        record.status = "progressing"
        record.reason = (
            f"value {direction}d in the direction the target requests"
        )
    else:
        record.status = "changed"
        record.reason = (
            f"value {direction}d, which is not the direction the target "
            "requests; review before drawing a conclusion"
        )
    return record


def build_tracking(
    earlier_findings: Sequence[VerifiedFinding],
    later_findings: Sequence[VerifiedFinding],
    config: dict[str, Any] | None = None,
    *,
    min_level: int | None = None,
    earlier_label: str = "earlier report",
    later_label: str = "later report",
) -> TrackingReport:
    """Compare quantified findings from two audits for the same targets."""
    cfg = config or {}
    tracking_cfg = get(cfg, "tracking", default={}) or {}
    level = coerce_rubric_level(
        min_level
        if min_level is not None
        else tracking_cfg.get("min_level", 3),
        setting="tracking min_level",
    )
    require_same_unit = bool(tracking_cfg.get("require_same_unit", True))

    early = _representatives(earlier_findings, min_level=level)
    late = _representatives(later_findings, min_level=level)

    records: list[TrackingRecord] = []
    for target_id in sorted(set(early) | set(late), key=target_sort_key):
        if target_id in early and target_id in late:
            records.append(
                _compare(
                    early[target_id],
                    late[target_id],
                    target_text=early[target_id].target_text
                    or late[target_id].target_text,
                    require_same_unit=require_same_unit,
                )
            )
        elif target_id in early:
            item = early[target_id]
            records.append(
                TrackingRecord(
                    target_id=target_id,
                    target_goal=item.target_goal,
                    target_text=item.target_text,
                    status="not_found_in_later_report",
                    earlier=item,
                    reason=(
                        "not found in the later report; this may reflect "
                        "retrieval or reporting changes and needs review"
                    ),
                )
            )
        else:
            item = late[target_id]
            records.append(
                TrackingRecord(
                    target_id=target_id,
                    target_goal=item.target_goal,
                    target_text=item.target_text,
                    status="new_in_later_report",
                    later=item,
                    reason="target appears only in the later report",
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


def track_payloads(
    earlier_payload: dict[str, Any],
    later_payload: dict[str, Any],
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TrackingReport:
    """Compare two audit JSON payloads."""
    return build_tracking(
        findings_from_payload(earlier_payload),
        findings_from_payload(later_payload),
        config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def render_markdown(report: TrackingReport) -> str:
    """Render a neutral Markdown tracking summary."""
    lines = [
        "# Multi-year SDG tracking",
        "",
        f"- Earlier: {report.earlier_label}",
        f"- Later: {report.later_label}",
        f"- Minimum rubric level: {report.min_level}",
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
                "## Targets",
                "",
                "| Target | Status | Earlier | Later | Change | Reason |",
                "|---|---|---|---|---|---|",
            ]
        )
        for record in report.records:
            earlier = (
                _format_value(record.earlier) if record.earlier else "-"
            )
            later = _format_value(record.later) if record.later else "-"
            change = (
                f"{record.change:+g}"
                if record.change is not None
                else "-"
            )
            lines.append(
                f"| {record.target_id} | {record.status_text} | "
                f"{earlier} | {later} | {change} | {record.reason} |"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare quantified SDG findings across two audit JSON files."
    )
    parser.add_argument("earlier", help="Earlier audit JSON")
    parser.add_argument("later", help="Later audit JSON")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Write tracking JSON here")
    parser.add_argument(
        "--markdown", default=None, help="Write tracking Markdown here"
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=None,
        help="Minimum rubric level to compare (default: config tracking.min_level)",
    )
    parser.add_argument(
        "--earlier-label", default="earlier report", help="Earlier report label"
    )
    parser.add_argument(
        "--later-label", default="later report", help="Later report label"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        earlier_payload = load_payload(args.earlier)
        later_payload = load_payload(args.later)
        report = track_payloads(
            earlier_payload,
            later_payload,
            config,
            min_level=args.min_level,
            earlier_label=args.earlier_label,
            later_label=args.later_label,
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
        print(f"Wrote tracking JSON: {path}")
    if args.markdown:
        path = resolve_path(args.markdown, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"Wrote tracking Markdown: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
