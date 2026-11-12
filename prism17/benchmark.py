"""Build and score a human-annotation benchmark for Prism17 findings.

This module does two things:

* :func:`build_annotation_rows` turns an audit JSON into a CSV template with
  blank reviewer columns. It does not invent labels.
* :func:`score_annotations` scores a filled template against the model output
  using the deterministic metrics in :mod:`prism17.evaluation`.

The benchmark measures disclosure-level agreement: relevance, rubric level,
quote verification, and semantic support. It does not measure real-world
sustainability performance. A template with no reviewer labels produces a
report with ``status: "no_annotations"`` and null metrics, never placeholder
numbers. Invalid reviewer values produce ``status: "invalid_annotations"``, and
reviewer values without a model counterpart produce ``status: "no_pairs"``;
both also leave the metrics null.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evaluation import binary_metrics, categorical_agreement, rubric_agreement
from .models import VerifiedFinding, is_not_relevant
from .tracking import infer_reporting_year

ANNOTATION_FIELDS = [
    "annotation_id",
    "audit_path",
    "company",
    "reporting_year",
    "pdf_path",
    "page",
    "passage_id",
    "target_id",
    "claim",
    "quote",
    "model_relevant",
    "model_level",
    "model_quote_verified",
    "model_semantic_support",
    "model_review_reason",
    "reviewer_relevant",
    "reviewer_level",
    "reviewer_quote_ok",
    "reviewer_semantic_support",
    "reviewer_notes",
    "annotator",
    "annotated_at_utc",
]

REQUIRED_COLUMNS = {
    "annotation_id",
    "page",
    "target_id",
    "claim",
    "quote",
    "model_relevant",
    "model_level",
    "reviewer_relevant",
    "reviewer_level",
}

LEVELS = (1, 2, 3, 4)
SEMANTIC_LABELS = ("supported", "uncertain", "not_supported")


def _label(document: dict[str, Any], fallback: str = "") -> str:
    title = str(document.get("title") or "").strip()
    if title:
        return title
    pdf_path = str(document.get("pdf_path") or "").strip()
    if pdf_path:
        return Path(pdf_path).stem
    return fallback


def _findings_from_payload(payload: dict[str, Any]) -> list[VerifiedFinding]:
    if "findings" not in payload:
        raise ValueError("audit JSON has no 'findings' list")
    return [
        VerifiedFinding.from_dict(item)
        for item in (payload.get("findings") or [])
    ]


def build_annotation_rows(
    payload: dict[str, Any],
    *,
    audit_path: str = "",
    prefix: str = "ann",
    include_unscored: bool = True,
    include_not_relevant: bool = False,
) -> list[dict[str, str]]:
    """Build blank-reviewer annotation rows from one audit payload.

    Scored findings and findings that need human review are included by
    default. Clean "judged not relevant" candidates are excluded unless
    ``include_not_relevant`` is set, because they are not claims about a target.
    """
    document = payload.get("document") or {}
    company = _label(document)
    year = ""
    inferred_year = infer_reporting_year(document)
    if inferred_year is not None:
        year = str(inferred_year)

    rows: list[dict[str, str]] = []
    for finding in _findings_from_payload(payload):
        if not include_unscored and not finding.is_scored:
            continue
        if is_not_relevant(finding) and not include_not_relevant:
            continue

        rows.append(
            {
                "annotation_id": f"{prefix}-{len(rows) + 1:04d}",
                "audit_path": str(audit_path),
                "company": company,
                "reporting_year": year,
                "pdf_path": str(document.get("pdf_path") or ""),
                "page": str(finding.source_page),
                "passage_id": finding.judged.candidate.passage_id,
                "target_id": finding.judged.sdg_target,
                "claim": finding.judged.claim,
                "quote": finding.judged.supporting_quote,
                "model_relevant": "1" if finding.judged.relevant else "0",
                "model_level": (
                    str(finding.level) if finding.level is not None else ""
                ),
                "model_quote_verified": "1" if finding.quote_verified else "0",
                "model_semantic_support": finding.semantic_support,
                "model_review_reason": finding.review_reason or "",
                "reviewer_relevant": "",
                "reviewer_level": "",
                "reviewer_quote_ok": "",
                "reviewer_semantic_support": "",
                "reviewer_notes": "",
                "annotator": "",
                "annotated_at_utc": "",
            }
        )
    return rows


def write_annotation_csv(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write annotation rows with a stable column order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in ANNOTATION_FIELDS}
            )
    return target


def read_annotation_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a gold CSV, validating the required columns."""
    target = Path(path)
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "gold CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        return [
            {key: (value or "") for key, value in row.items()}
            for row in reader
        ]


def _parse_level(value: str) -> tuple[int | None, bool]:
    text = str(value or "").strip()
    if not text:
        return None, True
    try:
        numeric = float(text)
    except ValueError:
        return None, False
    if not numeric.is_integer():
        return None, False
    parsed = int(numeric)
    if parsed not in LEVELS:
        return None, False
    return parsed, True


def _parse_binary(value: str) -> tuple[int | None, bool]:
    text = str(value or "").strip().lower()
    if not text:
        return None, True
    if text in {"1", "true", "yes", "y"}:
        return 1, True
    if text in {"0", "false", "no", "n"}:
        return 0, True
    return None, False


def _parse_semantic(value: str) -> tuple[str | None, bool]:
    text = str(value or "").strip().lower()
    if not text or text in {"not_checked", "not checked"}:
        return None, True
    if text in SEMANTIC_LABELS:
        return text, True
    return None, False


def _level_counts(values: Sequence[int]) -> dict[str, int]:
    return {
        str(level): sum(1 for value in values if value == level)
        for level in LEVELS
    }


@dataclass
class _Pairs:
    reviewer_relevance: list[int]
    model_relevance: list[int]
    reviewer_levels: list[int]
    model_levels: list[int]
    reviewer_quote: list[int]
    model_quote: list[int]
    reviewer_semantic: list[str] = field(default_factory=list)
    model_semantic: list[str] = field(default_factory=list)
    invalid_values: int = 0
    valid_reviewer_values: int = 0
    missing_relevance: int = 0
    missing_levels: int = 0
    missing_quote: int = 0
    missing_semantic: int = 0


def _collect_pairs(rows: Sequence[dict[str, str]]) -> _Pairs:
    pairs = _Pairs([], [], [], [], [], [])
    for row in rows:
        model_relevant, model_ok = _parse_binary(row.get("model_relevant", ""))
        reviewer_relevant, reviewer_ok = _parse_binary(
            row.get("reviewer_relevant", "")
        )
        model_level, model_level_ok = _parse_level(
            row.get("model_level", "")
        )
        reviewer_level, reviewer_level_ok = _parse_level(
            row.get("reviewer_level", "")
        )
        model_quote, model_quote_ok = _parse_binary(
            row.get("model_quote_verified", "")
        )
        reviewer_quote, reviewer_quote_ok = _parse_binary(
            row.get("reviewer_quote_ok", "")
        )
        model_semantic, model_semantic_ok = _parse_semantic(
            row.get("model_semantic_support", "")
        )
        reviewer_semantic, reviewer_semantic_ok = _parse_semantic(
            row.get("reviewer_semantic_support", "")
        )

        if not all(
            (
                model_ok,
                reviewer_ok,
                model_level_ok,
                reviewer_level_ok,
                model_quote_ok,
                reviewer_quote_ok,
                model_semantic_ok,
                reviewer_semantic_ok,
            )
        ):
            pairs.invalid_values += 1

        pairs.valid_reviewer_values += sum(
            value is not None
            for value in (
                reviewer_relevant,
                reviewer_level,
                reviewer_quote,
                reviewer_semantic,
            )
        )

        if model_relevant is not None and reviewer_relevant is not None:
            pairs.reviewer_relevance.append(reviewer_relevant)
            pairs.model_relevance.append(model_relevant)
        elif model_ok and reviewer_ok:
            pairs.missing_relevance += 1

        if model_level is not None and reviewer_level is not None:
            pairs.reviewer_levels.append(reviewer_level)
            pairs.model_levels.append(model_level)
        elif model_level_ok and reviewer_level_ok:
            pairs.missing_levels += 1

        if model_quote is not None and reviewer_quote is not None:
            pairs.reviewer_quote.append(reviewer_quote)
            pairs.model_quote.append(model_quote)
        elif model_quote_ok and reviewer_quote_ok:
            pairs.missing_quote += 1

        if model_semantic is not None and reviewer_semantic is not None:
            pairs.reviewer_semantic.append(reviewer_semantic)
            pairs.model_semantic.append(model_semantic)
        elif model_semantic_ok and reviewer_semantic_ok:
            pairs.missing_semantic += 1
    return pairs


def score_annotations(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Score a filled gold CSV.

    Each metric is computed only from rows where both the model value and the
    reviewer value are present and valid. When no reviewer labels are present,
    the matching metric is ``None`` and the overall status is
    ``no_annotations``. Invalid labels yield ``invalid_annotations``, and
    labels with no model counterpart yield ``no_pairs``.
    """
    pairs = _collect_pairs(rows)
    relevance_available = bool(pairs.reviewer_relevance)
    rubric_available = bool(pairs.reviewer_levels)
    quote_available = bool(pairs.reviewer_quote)
    semantic_available = bool(pairs.reviewer_semantic)
    metrics_available = (
        relevance_available
        or rubric_available
        or quote_available
        or semantic_available
    )
    if metrics_available:
        status = "ok"
    elif pairs.invalid_values:
        status = "invalid_annotations"
    elif pairs.valid_reviewer_values:
        status = "no_pairs"
    else:
        status = "no_annotations"

    return {
        "schema_version": "0.1",
        "status": status,
        "rows_total": len(rows),
        "reviewer_values_present": pairs.valid_reviewer_values,
        "pairs": {
            "relevance": len(pairs.reviewer_relevance),
            "rubric": len(pairs.reviewer_levels),
            "quote_verification": len(pairs.reviewer_quote),
            "semantic_support": len(pairs.reviewer_semantic),
        },
        "skipped": {
            "invalid_values": pairs.invalid_values,
            "missing_relevance": pairs.missing_relevance,
            "missing_levels": pairs.missing_levels,
            "missing_quote": pairs.missing_quote,
            "missing_semantic": pairs.missing_semantic,
        },
        "relevance": (
            binary_metrics(
                pairs.reviewer_relevance, pairs.model_relevance
            )
            if relevance_available
            else None
        ),
        "rubric": (
            rubric_agreement(pairs.reviewer_levels, pairs.model_levels)
            if rubric_available
            else None
        ),
        "quote_verification": (
            binary_metrics(pairs.reviewer_quote, pairs.model_quote)
            if quote_available
            else None
        ),
        "semantic_support": (
            categorical_agreement(
                pairs.reviewer_semantic,
                pairs.model_semantic,
                labels=SEMANTIC_LABELS,
            )
            if semantic_available
            else None
        ),
        "level_counts": {
            "reviewer": _level_counts(pairs.reviewer_levels),
            "model": _level_counts(pairs.model_levels),
        },
    }
