"""Human review decisions for the Phase 4 UI and reviewed reports.

A review decision is stored separately from the audit output, keyed by the
finding's passage, target, and page. The rubric level is never overwritten:
an override records a reviewer level alongside it so the original assessment
remains auditable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .models import VerifiedFinding

LOGGER = logging.getLogger(__name__)

REVIEW_STATUSES = ("pending", "confirmed", "overridden", "flagged")
SCHEMA_VERSION = "0.1"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def finding_key(finding: VerifiedFinding) -> str:
    """Stable identity for a finding across report reruns."""
    candidate = finding.judged.candidate
    return (
        f"{candidate.passage_id}|{finding.judged.sdg_target}|"
        f"{finding.source_page}"
    )


@dataclass
class ReviewDecision:
    """One reviewer decision about one scored finding."""

    finding_key: str
    status: str = "pending"
    level: int | None = None
    note: str = ""
    reviewer: str = ""
    updated_at: str = ""

    def normalised(self) -> "ReviewDecision":
        status = self.status if self.status in REVIEW_STATUSES else "pending"
        level = self.level if status == "overridden" else None
        if level is not None and level not in (1, 2, 3, 4):
            level = None
        return ReviewDecision(
            finding_key=self.finding_key,
            status=status,
            level=level,
            note=str(self.note or "").strip(),
            reviewer=str(self.reviewer or "").strip(),
            updated_at=self.updated_at or _now(),
        )

    def is_valid_override(self) -> bool:
        return self.status != "overridden" or self.level in (1, 2, 3, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_key": self.finding_key,
            "status": self.status,
            "level": self.level,
            "note": self.note,
            "reviewer": self.reviewer,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewDecision":
        key = data.get("finding_key") or data.get("key") or ""
        level = data.get("level")
        return cls(
            finding_key=str(key),
            status=str(data.get("status", "pending")),
            level=int(level) if level not in (None, "") else None,
            note=str(data.get("note", "")),
            reviewer=str(data.get("reviewer", "")),
            updated_at=str(data.get("updated_at", "")),
        )


@dataclass
class ReviewSummary:
    """Counts of review states across scored findings."""

    total: int = 0
    pending: int = 0
    confirmed: int = 0
    overridden: int = 0
    flagged: int = 0

    @property
    def reviewed(self) -> int:
        return self.confirmed + self.overridden + self.flagged

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "pending": self.pending,
            "confirmed": self.confirmed,
            "overridden": self.overridden,
            "flagged": self.flagged,
            "reviewed": self.reviewed,
        }


def load_reviews(path: str | Path) -> dict[str, ReviewDecision]:
    """Load decisions from disk, returning an empty mapping on any problem."""
    review_path = Path(path).expanduser()
    if not review_path.exists():
        return {}

    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read review file %s: %s", review_path, exc)
        return {}

    items: Any
    if isinstance(payload, dict):
        items = payload.get("reviews", [])
    elif isinstance(payload, list):
        items = payload
    else:
        LOGGER.warning("Unexpected review file format: %s", review_path)
        return {}

    decisions: dict[str, ReviewDecision] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = ReviewDecision.from_dict(item).normalised()
        if decision.finding_key:
            decisions[decision.finding_key] = decision
    return decisions


def save_reviews(
    path: str | Path,
    decisions: dict[str, ReviewDecision],
) -> Path:
    """Write decisions to disk in a stable, reviewable JSON format."""
    review_path = Path(path).expanduser()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "reviews": [
            decisions[key].normalised().to_dict()
            for key in sorted(decisions)
        ],
    }
    review_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return review_path


def make_decision(
    key: str,
    *,
    status: str,
    level: int | None = None,
    note: str = "",
    reviewer: str = "",
) -> ReviewDecision:
    return ReviewDecision(
        finding_key=key,
        status=status,
        level=level,
        note=note,
        reviewer=reviewer,
        updated_at=_now(),
    ).normalised()


def summarise_reviews(
    findings: Sequence[VerifiedFinding],
    reviews: dict[str, ReviewDecision],
) -> ReviewSummary:
    """Count review states for scored findings only."""
    summary = ReviewSummary()
    for finding in findings:
        if not finding.is_scored:
            continue
        summary.total += 1
        decision = reviews.get(finding_key(finding))
        status = decision.status if decision else "pending"
        if status == "confirmed":
            summary.confirmed += 1
        elif status == "overridden":
            summary.overridden += 1
        elif status == "flagged":
            summary.flagged += 1
        else:
            summary.pending += 1
    return summary


def export_rows(
    findings: Sequence[VerifiedFinding],
    reviews: dict[str, ReviewDecision],
) -> list[dict[str, Any]]:
    """Flatten scored findings and decisions for CSV or report tables."""
    rows: list[dict[str, Any]] = []
    for finding in findings:
        if not finding.is_scored:
            continue
        key = finding_key(finding)
        decision = reviews.get(key)
        rows.append(
            {
                "finding_key": key,
                "goal": finding.judged.sdg_goal,
                "target": finding.judged.sdg_target,
                "page": finding.source_page,
                "rubric_level": finding.level,
                "review_status": decision.status if decision else "pending",
                "review_level": decision.level if decision else None,
                "note": decision.note if decision else "",
                "reviewer": decision.reviewer if decision else "",
                "updated_at": decision.updated_at if decision else "",
                "quote": finding.judged.supporting_quote,
            }
        )
    return rows
