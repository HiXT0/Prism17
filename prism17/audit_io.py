"""Small helpers for reading Prism17 audit JSON files.

Keeping payload validation in one place ensures tracking, comparison, and
benchmark tooling reject malformed audits consistently instead of silently
treating a missing findings list as an empty report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import VerifiedFinding

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load one audit JSON object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"audit JSON must be an object: {path}")
    return payload


def findings_from_payload(payload: dict[str, Any]) -> list[VerifiedFinding]:
    """Return findings from an audit payload, rejecting absent/invalid lists."""
    if "findings" not in payload:
        raise ValueError("audit JSON has no 'findings' list")
    raw = payload.get("findings")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("audit JSON 'findings' must be a list")
    findings: list[VerifiedFinding] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"audit finding {index} is not an object")
        findings.append(VerifiedFinding.from_dict(item))
    return findings


def infer_reporting_year(payload: dict[str, Any]) -> int | None:
    """Infer the report year from document metadata or extracted values."""
    document = payload.get("document")
    if isinstance(document, dict):
        for key in ("year", "reporting_year", "report_year", "period", "title"):
            value = document.get(key)
            match = _YEAR_RE.search(str(value or ""))
            if match:
                return int(match.group(1))

    for item in payload.get("findings") or []:
        if not isinstance(item, dict):
            continue
        judged = item.get("judged")
        if not isinstance(judged, dict):
            continue
        extracted = judged.get("extracted")
        if not isinstance(extracted, dict):
            continue
        match = _YEAR_RE.search(str(extracted.get("year", "") or ""))
        if match:
            return int(match.group(1))
    return None
