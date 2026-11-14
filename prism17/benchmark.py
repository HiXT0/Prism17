"""Human-annotation benchmark tooling.

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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .audit_io import findings_from_payload, infer_reporting_year
from .evaluation import (
    binary_metrics,
    categorical_agreement,
    confusion_matrix,
    rubric_agreement,
)
from .models import VerifiedFinding

LEVELS = (1, 2, 3, 4)
SEMANTIC_LABELS = ("supported", "uncertain", "not_supported")

BENCHMARK_FIELDS = [
    "annotation_id",
    "company",
    "year",
    "source_path",
    "title",
    "target_id",
    "page",
    "passage_id",
    "claim",
    "quote",
    "model_relevant",
    "model_level",
    "model_quote_ok",
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
    "claim",
    "page",
    "quote",
    "model_relevant",
    "model_level",
    "model_quote_ok",
    "model_semantic_support",
    "reviewer_relevant",
    "reviewer_level",
    "reviewer_quote_ok",
    "reviewer_semantic_support",
}

_TRUE_VALUES = {"1", "true", "yes", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "n", "f"}


@dataclass
class PairCollector:
    reviewer_relevance: list[int] = field(default_factory=list)
    model_relevance: list[int] = field(default_factory=list)
    reviewer_levels: list[int] = field(default_factory=list)
    model_levels: list[int] = field(default_factory=list)
    reviewer_quote: list[int] = field(default_factory=list)
    model_quote: list[int] = field(default_factory=list)
    reviewer_semantic: list[str] = field(default_factory=list)
    model_semantic: list[str] = field(default_factory=list)
    invalid_values: int = 0
    invalid_model_values: int = 0
    valid_reviewer_values: int = 0
    missing_relevance: int = 0
    missing_levels: int = 0
    missing_quote: int = 0
    missing_semantic: int = 0


def parse_bool(value: Any) -> bool | None:
    """Parse a boolean annotation value; blank returns ``None``."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_level(value: Any) -> int | None:
    """Parse a strict integer rubric level; fractional values are invalid."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid rubric level: {value!r}") from exc
    if not numeric.is_integer():
        raise ValueError(f"invalid rubric level: {value!r}")
    level = int(numeric)
    if level not in LEVELS:
        raise ValueError(f"invalid rubric level: {value!r}")
    return level


def parse_semantic(value: Any) -> str | None:
    """Parse a semantic-support label; blank and ``not_checked`` return None."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text == "not_checked":
        return None
    if text not in SEMANTIC_LABELS:
        raise ValueError(f"invalid semantic-support value: {value!r}")
    return text


def template_rows(
    payloads: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build annotation-template rows from one or more audit payloads."""
    if labels and len(labels) != len(payloads):
        raise ValueError("labels must match the number of audit files")
    if paths and len(paths) != len(payloads):
        raise ValueError("paths must match the number of audit files")

    rows: list[dict[str, Any]] = []
    for payload_index, payload in enumerate(payloads):
        path = paths[payload_index] if paths else ""
        fallback = Path(path).stem if path else f"audit-{payload_index + 1}"
        label = labels[payload_index] if labels else fallback
        document = payload.get("document")
        title = (
            str(document.get("title") or "")
            if isinstance(document, dict)
            else ""
        ) or label
        year = infer_reporting_year(payload)
        findings: list[VerifiedFinding] = findings_from_payload(payload)
        for finding_index, finding in enumerate(findings, start=1):
            rows.append(
                {
                    "annotation_id": (
                        f"ann-{Path(path).stem if path else fallback}-"
                        f"{finding_index:04d}"
                    ),
                    "company": label,
                    "year": year or "",
                    "source_path": path,
                    "title": title,
                    "target_id": finding.judged.sdg_target,
                    "page": finding.source_page,
                    "passage_id": finding.judged.candidate.passage_id,
                    "claim": finding.judged.claim,
                    "quote": finding.judged.supporting_quote,
                    "model_relevant": int(bool(finding.judged.relevant)),
                    "model_level": (
                        "" if finding.level is None else int(finding.level)
                    ),
                    "model_quote_ok": int(bool(finding.quote_verified)),
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


def write_template(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    *,
    force: bool = False,
) -> Path:
    """Write a CSV annotation template, refusing to overwrite by default."""
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(
            f"Gold file already exists: {target}. Use --force to overwrite."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a gold CSV and validate that required columns are present."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise ValueError(
                "gold CSV is missing required columns: " + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def _try_parse(parser: Any, value: Any) -> tuple[Any, bool]:
    try:
        return parser(value), True
    except ValueError:
        return None, False


def _collect_pairs(rows: Sequence[dict[str, Any]]) -> PairCollector:
    pairs = PairCollector()
    for row in rows:
        model_relevant, model_relevant_ok = _try_parse(
            parse_bool, row.get("model_relevant")
        )
        model_level, model_level_ok = _try_parse(
            parse_level, row.get("model_level")
        )
        model_quote, model_quote_ok = _try_parse(
            parse_bool, row.get("model_quote_ok")
        )
        model_semantic, model_semantic_ok = _try_parse(
            parse_semantic, row.get("model_semantic_support")
        )

        reviewer_relevant, reviewer_relevant_ok = _try_parse(
            parse_bool, row.get("reviewer_relevant")
        )
        reviewer_level, reviewer_level_ok = _try_parse(
            parse_level, row.get("reviewer_level")
        )
        reviewer_quote, reviewer_quote_ok = _try_parse(
            parse_bool, row.get("reviewer_quote_ok")
        )
        reviewer_semantic, reviewer_semantic_ok = _try_parse(
            parse_semantic, row.get("reviewer_semantic_support")
        )

        if not all(
            (
                model_relevant_ok,
                model_level_ok,
                model_quote_ok,
                model_semantic_ok,
            )
        ):
            pairs.invalid_model_values += 1
        if not all(
            (
                reviewer_relevant_ok,
                reviewer_level_ok,
                reviewer_quote_ok,
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

        if reviewer_relevant is not None and model_relevant is not None:
            pairs.reviewer_relevance.append(int(bool(reviewer_relevant)))
            pairs.model_relevance.append(int(bool(model_relevant)))
        elif reviewer_relevant is not None:
            pairs.missing_relevance += 1

        if reviewer_level is not None and model_level is not None:
            pairs.reviewer_levels.append(int(reviewer_level))
            pairs.model_levels.append(int(model_level))
        elif reviewer_level is not None:
            pairs.missing_levels += 1

        if reviewer_quote is not None and model_quote is not None:
            pairs.reviewer_quote.append(int(bool(reviewer_quote)))
            pairs.model_quote.append(int(bool(model_quote)))
        elif reviewer_quote is not None:
            pairs.missing_quote += 1

        if (
            reviewer_semantic is not None
            and model_semantic is not None
            and model_semantic != "not_checked"
        ):
            pairs.reviewer_semantic.append(str(reviewer_semantic))
            pairs.model_semantic.append(str(model_semantic))
        elif reviewer_semantic is not None:
            pairs.missing_semantic += 1

    return pairs


def _binary_block(
    reviewer: Sequence[int],
    model: Sequence[int],
) -> dict[str, Any] | None:
    if not reviewer:
        return None
    metrics = binary_metrics(reviewer, model)
    metrics["confusion"] = confusion_matrix(reviewer, model, [0, 1])
    return metrics


def score_annotations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score reviewer labels against model labels without fabricating values."""
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
            "invalid_model_values": pairs.invalid_model_values,
            "missing_relevance": pairs.missing_relevance,
            "missing_levels": pairs.missing_levels,
            "missing_quote": pairs.missing_quote,
            "missing_semantic": pairs.missing_semantic,
        },
        "relevance": _binary_block(
            pairs.reviewer_relevance, pairs.model_relevance
        ),
        "rubric": (
            rubric_agreement(pairs.reviewer_levels, pairs.model_levels)
            if rubric_available
            else None
        ),
        "quote_verification": _binary_block(
            pairs.reviewer_quote, pairs.model_quote
        ),
        "semantic_support": (
            categorical_agreement(
                pairs.reviewer_semantic,
                pairs.model_semantic,
                SEMANTIC_LABELS,
            )
            if semantic_available
            else None
        ),
    }


def score_csv(path: str | Path) -> dict[str, Any]:
    """Read a gold CSV and return benchmark metrics."""
    return score_annotations(read_rows(path))


def write_report(path: str | Path, document: dict[str, Any]) -> Path:
    """Write a benchmark JSON report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
