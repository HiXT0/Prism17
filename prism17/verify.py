"""Stage 1 quote verification.

The judge is asked to copy an exact substring from the passage. This module
checks that claim against the source page before any finding can be scored.

Normalisation is deliberately narrow: we collapse whitespace, repair PDF
line-break hyphenation, and unify typographic quotes/dashes. We do not reorder
words, remove punctuation, or paraphrase. An exact match after normalisation is
accepted; otherwise a case-sensitive fuzzy partial match must clear the
configured threshold (default 95/100).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

from .config import get
from .models import JudgedFinding, VerifiedFinding
from .parse import ParsedDocument

LOGGER = logging.getLogger(__name__)

_INVISIBLE_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff]")
_HYPHEN_BREAK_RE = re.compile(
    r"(?<=[A-Za-z])[-\u2010\u2011]\s*\n\s*(?=[a-z])"
)
_WHITESPACE_RE = re.compile(r"\s+")

# Map typographic variants to a single form so a model's smart quotes do not
# cause a false verification failure. Applied to both quote and source.
_CHAR_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


@dataclass
class QuoteCheck:
    """Result of checking one quote against one source page."""

    verified: bool
    method: str | None
    score: float
    matched_text: str = ""
    reason: str | None = None


def normalise_for_match(text: str) -> str:
    """Normalise text for quote matching without changing word order."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub("", text)
    text = text.translate(_CHAR_MAP)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _fuzzy_ratio(quote: str, source: str) -> float:
    """Return RapidFuzz partial ratio, or 0 when the dependency is absent."""
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - depends on environment
        LOGGER.warning(
            "rapidfuzz is not installed; fuzzy quote verification is disabled"
        )
        return 0.0
    return float(fuzz.partial_ratio(quote, source))


def _check_against(
    quote_norm: str,
    source_text: str,
    threshold: float,
    *,
    label: str,
) -> QuoteCheck | None:
    source_norm = normalise_for_match(source_text)
    if not source_norm:
        return None

    if quote_norm in source_norm:
        return QuoteCheck(
            verified=True,
            method=f"{label}_exact",
            score=100.0,
            matched_text=quote_norm,
        )

    score = _fuzzy_ratio(quote_norm, source_norm)
    if score >= threshold:
        return QuoteCheck(
            verified=True,
            method=f"{label}_fuzzy",
            score=round(score, 2),
            matched_text=quote_norm,
        )
    return QuoteCheck(
        verified=False,
        method=f"{label}_fuzzy",
        score=round(score, 2),
        reason=(
            f"quote not found on the {label} "
            f"(best fuzzy score {score:.1f} < {threshold:.1f})"
        ),
    )


def verify_quote(
    quote: str,
    *,
    page_text: str | None,
    passage_text: str | None,
    threshold: float = 95.0,
) -> QuoteCheck:
    """Check a quote against the raw page, then the passage text.

    Page text is preferred because it is the strongest evidence source. The
    passage fallback handles table passages, whose pipe-separated rendering
    differs from the raw page text.
    """
    quote_norm = normalise_for_match(quote)
    if not quote_norm:
        return QuoteCheck(
            verified=False,
            method=None,
            score=0.0,
            reason="quote is empty after normalisation",
        )
    if len(quote_norm.split()) < 4:
        return QuoteCheck(
            verified=False,
            method=None,
            score=0.0,
            reason="quote is too short to verify",
        )

    best_failure: QuoteCheck | None = None
    for label, text in (("page", page_text), ("passage", passage_text)):
        if not text:
            continue
        result = _check_against(quote_norm, text, threshold, label=label)
        if result is None:
            continue
        if result.verified:
            return result
        if best_failure is None or result.score > best_failure.score:
            best_failure = result

    return best_failure or QuoteCheck(
        verified=False,
        method=None,
        score=0.0,
        reason="no source text available for this page",
    )


def verify_finding(
    finding: JudgedFinding,
    document: ParsedDocument,
    *,
    threshold: float = 95.0,
) -> VerifiedFinding:
    """Verify one judged finding and attach its source page.

    A finding that the judge marked irrelevant, or that already failed schema
    validation, is passed through as unscored. It is never silently promoted.
    """
    candidate = finding.candidate
    verified = VerifiedFinding(
        judged=finding,
        quote_verified=False,
        quote_method=None,
        quote_score=0.0,
        source_page=candidate.page,
        review_reason=finding.review_reason,
    )

    if finding.review_reason is not None:
        return verified
    if not finding.relevant:
        verified.review_reason = "judge marked the passage as not relevant"
        return verified

    check = verify_quote(
        finding.supporting_quote,
        page_text=document.page_text(candidate.page),
        passage_text=candidate.passage_text,
        threshold=threshold,
    )
    verified.quote_verified = check.verified
    verified.quote_method = check.method
    verified.quote_score = check.score
    if not check.verified:
        verified.review_reason = (
            "unscored (needs human review): quote could not be verified: "
            f"{check.reason or 'unknown reason'}"
        )
        LOGGER.info(
            "Quote verification failed for %s page %s: %s",
            candidate.passage_id,
            candidate.page,
            check.reason,
        )
    return verified


def verify_findings(
    findings: Sequence[JudgedFinding],
    document: ParsedDocument,
    config: dict[str, Any] | None = None,
) -> list[VerifiedFinding]:
    """Verify a sequence of judged findings using the configured threshold."""
    cfg = config or {}
    threshold = float(get(cfg, "verification", "fuzzy_threshold", default=95))
    return [
        verify_finding(finding, document, threshold=threshold)
        for finding in findings
    ]
