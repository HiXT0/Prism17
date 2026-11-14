"""Stage 2 semantic support verification.

Stage 1 checks that a supporting quote exists on the source page. Stage 2 asks
whether that quote actually supports the judge's claim. It is deliberately
conservative and opt-in:

* ``supported``: the claim is entailed by the quote.
* ``uncertain``: the evidence is ambiguous; a human should review it.
* ``not_supported``: the quote does not support the claim.
* ``not_checked``: Stage 2 was disabled or Stage 1 had already failed.

A deterministic guard rejects claims that share no content words with the
quote. The configured method then uses either a local NLI cross-encoder or the
local Ollama model. If the NLI model cannot be loaded, the finding becomes
``uncertain`` rather than ``supported``; it is never silently scored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .config import get
from .judge import SupportJudge
from .models import VerifiedFinding

LOGGER = logging.getLogger(__name__)

SUPPORTED = "supported"
UNCERTAIN = "uncertain"
NOT_SUPPORTED = "not_supported"
NOT_CHECKED = "not_checked"

BLOCKING_SUPPORT = {NOT_SUPPORTED, UNCERTAIN}

_CROSS_ENCODER_CACHE: dict[str, Any] = {}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "had",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "we",
    "will",
    "with",
}


@dataclass
class SemanticCheck:
    """Result of comparing one claim with one supporting quote."""

    support: str
    score: float | None
    method: str
    reason: str | None = None


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def deterministic_check(claim: str, quote: str) -> SemanticCheck | None:
    """Return a definitive deterministic result, or ``None`` for NLI.

    The guard only rejects. Lexical overlap is not treated as proof of
    semantic support.
    """
    if not str(claim or "").strip() or not str(quote or "").strip():
        return SemanticCheck(
            NOT_SUPPORTED,
            0.0,
            "deterministic",
            "claim or quote is empty",
        )

    claim_tokens = _content_tokens(claim)
    quote_tokens = _content_tokens(quote)
    if not claim_tokens or not quote_tokens:
        return None
    if claim_tokens.isdisjoint(quote_tokens):
        return SemanticCheck(
            NOT_SUPPORTED,
            0.0,
            "deterministic",
            "claim and quote share no content words",
        )
    return None


def _fallback_check(claim: str, quote: str, reason: str) -> SemanticCheck:
    guard = deterministic_check(claim, quote)
    if guard is not None:
        return guard
    claim_tokens = _content_tokens(claim)
    quote_tokens = _content_tokens(quote)
    union = claim_tokens | quote_tokens
    overlap = len(claim_tokens & quote_tokens) / len(union) if union else 0.0
    return SemanticCheck(
        UNCERTAIN,
        round(overlap, 4),
        "deterministic_fallback",
        f"semantic model unavailable ({reason}); needs human review",
    )


def _load_cross_encoder(model_name: str) -> Any:
    if model_name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[model_name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "sentence-transformers is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    LOGGER.info("Loading NLI cross-encoder on CPU: %s", model_name)
    model = CrossEncoder(model_name, device="cpu")
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


def _entailment_index(model: Any) -> int:
    config = getattr(model, "config", None)
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if str(label).strip().lower() == "entailment":
            return int(index)
    # The documented order for cross-encoder/nli-deberta-v3-small.
    return 1


def _predict_probs(model: Any, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
    try:
        predictions = model.predict(
            list(pairs),
            apply_softmax=True,
            convert_to_numpy=True,
        )
    except TypeError:  # older sentence-transformers versions
        predictions = model.predict(list(pairs), convert_to_numpy=True)

    rows: list[list[float]] = []
    for row in predictions:
        values = [float(value) for value in row]
        total = sum(values)
        if total > 0 and not _looks_like_probabilities(values):
            values = [value / total for value in values]
        rows.append(values)
    return rows


def _looks_like_probabilities(values: Sequence[float]) -> bool:
    return all(0.0 <= value <= 1.0 for value in values) and abs(
        sum(values) - 1.0
    ) < 0.05


def nli_check(
    claim: str,
    quote: str,
    config: dict[str, Any],
    *,
    model: Any | None = None,
) -> SemanticCheck:
    """Score claim/quote support with the local NLI cross-encoder."""
    guard = deterministic_check(claim, quote)
    if guard is not None:
        return guard

    cfg = get(config, "verification", default={}) or {}
    model_name = str(
        cfg.get("nli_model", "cross-encoder/nli-deberta-v3-small")
    )
    supported_threshold = float(cfg.get("nli_supported_threshold", 0.60))
    uncertain_threshold = float(cfg.get("nli_uncertain_threshold", 0.35))

    try:
        active_model = model or _load_cross_encoder(model_name)
        probabilities = _predict_probs(active_model, [(quote, claim)])[0]
        index = _entailment_index(active_model)
        if index >= len(probabilities):
            raise ValueError("model output has no entailment column")
        score = float(probabilities[index])
    except Exception as exc:  # noqa: BLE001 - model/runtime availability
        LOGGER.warning("NLI semantic check failed: %s", exc)
        return _fallback_check(claim, quote, str(exc))

    if score >= supported_threshold:
        support = SUPPORTED
    elif score >= uncertain_threshold:
        support = UNCERTAIN
    else:
        support = NOT_SUPPORTED
    return SemanticCheck(
        support,
        round(score, 4),
        "nli",
        None if support == SUPPORTED else f"entailment score {score:.3f}",
    )


def llm_check(
    claim: str,
    quote: str,
    config: dict[str, Any],
    *,
    judge: SupportJudge | None = None,
) -> SemanticCheck:
    """Score claim/quote support with the local Ollama model."""
    guard = deterministic_check(claim, quote)
    if guard is not None:
        return guard

    active_judge = judge or SupportJudge(config)
    judgement = active_judge.judge(claim, quote)
    if not judgement.ok:
        return SemanticCheck(
            UNCERTAIN,
            None,
            "llm",
            judgement.error or "local support judge failed",
        )
    mapping = {
        "yes": SUPPORTED,
        "no": NOT_SUPPORTED,
        "unclear": UNCERTAIN,
    }
    support = mapping.get(judgement.supported, UNCERTAIN)
    return SemanticCheck(
        support,
        judgement.confidence,
        "llm",
        judgement.reason or None,
    )


def check_semantics(
    claim: str,
    quote: str,
    config: dict[str, Any],
    *,
    method: str | None = None,
    model: Any | None = None,
    judge: SupportJudge | None = None,
) -> SemanticCheck:
    """Run the configured Stage 2 method for one claim/quote pair."""
    cfg = get(config, "verification", default={}) or {}
    active_method = str(method or cfg.get("semantic_method", "nli")).lower()
    if active_method == "nli":
        return nli_check(claim, quote, config, model=model)
    if active_method == "llm":
        return llm_check(claim, quote, config, judge=judge)
    raise ValueError(
        f"Unknown semantic method {active_method!r}; use 'nli' or 'llm'"
    )


def verify_semantics(
    findings: Sequence[VerifiedFinding],
    config: dict[str, Any] | None = None,
    *,
    enabled: bool | None = None,
    method: str | None = None,
    model: Any | None = None,
    judge: SupportJudge | None = None,
) -> list[VerifiedFinding]:
    """Attach Stage 2 support to findings that passed Stage 1.

    Findings that fail semantic support or are uncertain receive a review
    reason. They are never scored. When Stage 2 is disabled, every finding is
    marked ``not_checked`` and existing review reasons are left untouched.
    """
    cfg = config or {}
    verification_cfg = get(cfg, "verification", default={}) or {}
    active = (
        bool(verification_cfg.get("semantic_enabled", False))
        if enabled is None
        else bool(enabled)
    )

    if not active:
        for finding in findings:
            finding.semantic_support = NOT_CHECKED
            finding.semantic_score = None
        return list(findings)

    for finding in findings:
        if (
            not finding.judged.relevant
            or not finding.judged.scored
            or not finding.quote_verified
            or finding.review_reason is not None
        ):
            continue

        check = check_semantics(
            finding.judged.claim,
            finding.judged.supporting_quote,
            cfg,
            method=method,
            model=model,
            judge=judge,
        )
        finding.semantic_support = check.support
        finding.semantic_score = check.score
        if check.support in BLOCKING_SUPPORT:
            finding.review_reason = (
                "unscored (needs human review): semantic support "
                f"{check.support}"
                + (f": {check.reason}" if check.reason else "")
            )
            LOGGER.info(
                "Semantic support %s for %s page %s",
                check.support,
                finding.judged.sdg_target,
                finding.source_page,
            )
    return list(findings)
