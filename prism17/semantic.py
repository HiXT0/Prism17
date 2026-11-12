"""Stage 2 verification: does the quote actually support the claim?

Stage 1 (:mod:`prism17.verify`) checks that a quote exists on the cited page.
This module checks whether the quote semantically supports the claim the judge
attributed to it. It is deliberately conservative:

* A deterministic consistency guard compares negation, years, numbers, units
  and commitment status between the claim and the quote. A mismatch makes the
  finding *uncertain* rather than silently trusted.
* The semantic decision itself comes from a local NLI cross-encoder
  (``cross-encoder/nli-deberta-v3-small`` by default), or optionally from the
  local Ollama model. No remote API is used.
* If the model is unavailable, the finding is left ``not_checked`` and is not
  penalised. A pipeline must still work on a laptop that has not downloaded the
  NLI model yet.

Only ``not_supported`` and ``uncertain`` block scoring; ``not_checked`` does
not. That keeps the Stage 1 guarantee intact while making Stage 2 opt-in by
dependency availability.

Run:
    The check is applied automatically by ``python -m prism17.pipeline`` when
    ``verification.semantic_enabled`` is true. Use ``--no-semantic`` to skip it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import get
from .models import VerifiedFinding

LOGGER = logging.getLogger(__name__)

SEMANTIC_STATUSES = ("supported", "not_supported", "uncertain", "not_checked")
BLOCKING_STATUSES = frozenset({"not_supported", "uncertain"})

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|without|cannot|can't|doesn't|don't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|won't|"
    r"no longer|neither|nor)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(%)?",
)
_UNIT_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(%|(?:percent|per cent|tco2e|tco2|mtco2e|ktco2e|mwh|gwh|kwh|gj|"
    r"kl|m3|litres?|liters?|tonnes?|tons?|kg|inr|crore|lakh|million|billion)\b)",
    re.IGNORECASE,
)
_COMMITMENT_PATTERNS = (
    ("achieved", re.compile(
        r"\b(?:achieved|attained|met|completed|realised|realized|delivered|"
        r"reached|exceeded)\b",
        re.IGNORECASE,
    )),
    ("planned", re.compile(
        r"\b(?:will|shall|plan|plans|planned|target|targets|aim|aims|"
        r"intend|intends|commit|commits|expect|expects|propose|proposes|"
        r"by\s+(?:19|20)\d{2})\b",
        re.IGNORECASE,
    )),
    ("ongoing", re.compile(
        r"\b(?:ongoing|continues?|continuing|currently|in progress|"
        r"underway|under way)\b",
        re.IGNORECASE,
    )),
)
_UNIT_ALIASES = {
    "percent": "%",
    "per cent": "%",
    "percentage": "%",
    "tco2": "tco2e",
    "mtco2e": "tco2e",
    "ktco2e": "tco2e",
    "tonnes": "t",
    "tonne": "t",
    "tons": "t",
    "ton": "t",
    "litres": "l",
    "litre": "l",
    "liters": "l",
    "liter": "l",
    "m3": "m3",
    "kl": "kl",
}


@dataclass
class SemanticResult:
    """Outcome of one semantic-support check."""

    status: str
    score: float | None = None
    method: str = ""
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks_scoring(self) -> bool:
        return self.status in BLOCKING_STATUSES


def _normalise_unit(unit: str) -> str:
    value = " ".join(str(unit or "").strip().lower().split())
    if not value:
        return ""
    return _UNIT_ALIASES.get(value, value)


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(1).replace(",", "")
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _years(text: str) -> set[int]:
    return {int(match.group(1)) for match in _YEAR_RE.finditer(text or "")}


def _units(text: str) -> set[str]:
    found: set[str] = set()
    for match in _UNIT_RE.finditer(text or ""):
        found.add(_normalise_unit(match.group(2)))
    return {unit for unit in found if unit}


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text or ""))


def _commitment_status(text: str) -> str:
    lowered = text or ""
    for label, pattern in _COMMITMENT_PATTERNS:
        if pattern.search(lowered):
            return label
    return "unclear"


def _numeric_values_without_years(text: str) -> list[float]:
    years = _years(text)
    values: list[float] = []
    for value in _numbers(text):
        if value.is_integer() and 1900 <= value <= 2100 and int(value) in years:
            # The token is a calendar year, not a measured value.
            continue
        values.append(value)
    return values


def consistency_check(claim: str, quote: str) -> SemanticResult:
    """Compare claim and quote for negation, years, numbers, units, status.

    The check is intentionally one-directional: everything the *claim* states
    should be traceable to the quote. The quote may contain additional context.
    A mismatch returns ``uncertain`` so a human can review it; it never returns
    ``not_supported`` on its own.
    """
    claim_years = _years(claim)
    quote_years = _years(quote)
    missing_years = sorted(claim_years - quote_years)
    if missing_years:
        return SemanticResult(
            status="uncertain",
            method="consistency",
            reason=(
                "claim cites year(s) not present in the quote: "
                + ", ".join(str(year) for year in missing_years)
            ),
            details={"missing_years": missing_years},
        )

    claim_units = _units(claim)
    quote_units = _units(quote)
    if claim_units and quote_units and not (claim_units & quote_units):
        return SemanticResult(
            status="uncertain",
            method="consistency",
            reason=(
                "claim and quote use different units "
                f"({', '.join(sorted(claim_units))} vs "
                f"{', '.join(sorted(quote_units))})"
            ),
            details={"claim_units": sorted(claim_units), "quote_units": sorted(quote_units)},
        )

    claim_numbers = _numeric_values_without_years(claim)
    quote_numbers = set(_numeric_values_without_years(quote))
    missing_numbers = [
        value for value in claim_numbers if value not in quote_numbers
    ]
    if missing_numbers:
        return SemanticResult(
            status="uncertain",
            method="consistency",
            reason=(
                "claim cites value(s) not present in the quote: "
                + ", ".join(f"{value:g}" for value in missing_numbers)
            ),
            details={"missing_numbers": missing_numbers},
        )

    claim_negated = _has_negation(claim)
    quote_negated = _has_negation(quote)
    if claim_negated != quote_negated:
        return SemanticResult(
            status="uncertain",
            method="consistency",
            reason="negation differs between the claim and the quote",
            details={
                "claim_negated": claim_negated,
                "quote_negated": quote_negated,
            },
        )

    claim_status = _commitment_status(claim)
    quote_status = _commitment_status(quote)
    if (
        claim_status != "unclear"
        and quote_status != "unclear"
        and claim_status != quote_status
    ):
        return SemanticResult(
            status="uncertain",
            method="consistency",
            reason=(
                "commitment status differs between the claim and the quote "
                f"(claim: {claim_status}, quote: {quote_status})"
            ),
            details={
                "claim_status": claim_status,
                "quote_status": quote_status,
            },
        )

    return SemanticResult(status="supported", method="consistency")


def _entailment_index(model: Any) -> int:
    """Find the entailment label index in a cross-encoder's config."""
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if "entail" in str(label).lower():
            return int(index)
    # The default sentence-transformers NLI ordering used by the configured
    # model is contradiction, entailment, neutral.
    return 1


_NLI_CACHE: dict[str, Any] = {}


def _load_nli_model(model_name: str) -> Any:
    if model_name in _NLI_CACHE:
        return _NLI_CACHE[model_name]
    from sentence_transformers import CrossEncoder  # local import: optional dep

    LOGGER.info("Loading NLI cross-encoder on CPU: %s", model_name)
    model = CrossEncoder(model_name, device="cpu")
    _NLI_CACHE[model_name] = model
    return model


def _nli_entailment_probability(
    claim: str,
    quote: str,
    config: dict[str, Any],
) -> float:
    """Return the entailment probability for (quote premise, claim hypothesis)."""
    verification = get(config, "verification", default={}) or {}
    model_name = str(
        verification.get("nli_model", "cross-encoder/nli-deberta-v3-small")
    )
    model = _load_nli_model(model_name)
    scores = model.predict([(quote, claim)], apply_softmax=True)
    row = scores[0]
    try:
        probabilities = [float(value) for value in row]
    except TypeError:
        probabilities = [float(row)]
    index = _entailment_index(model)
    if index >= len(probabilities):
        raise ValueError(
            f"NLI model returned {len(probabilities)} scores but entailment "
            f"index is {index}"
        )
    return max(0.0, min(1.0, probabilities[index]))


def _status_from_score(score: float, config: dict[str, Any]) -> str:
    verification = get(config, "verification", default={}) or {}
    supported = float(verification.get("nli_supported_threshold", 0.60))
    uncertain = float(verification.get("nli_uncertain_threshold", 0.35))
    if score >= supported:
        return "supported"
    if score >= uncertain:
        return "uncertain"
    return "not_supported"


def _llm_support(
    claim: str,
    quote: str,
    config: dict[str, Any],
    *,
    client: Any | None = None,
) -> SemanticResult:
    """Ask the local Ollama model whether the quote supports the claim."""
    llm_cfg = get(config, "llm", default={}) or {}
    model = str(llm_cfg.get("model", "qwen2.5:7b-instruct"))
    if client is None:
        try:
            import ollama
        except ImportError:
            return SemanticResult(
                status="not_checked",
                method="llm",
                reason="the ollama Python package is not installed",
            )
        try:
            client = ollama.Client(
                host=str(llm_cfg.get("base_url", "http://localhost:11434")),
                timeout=float(llm_cfg.get("request_timeout_seconds", 300)),
            )
        except TypeError:
            client = ollama.Client(
                host=str(llm_cfg.get("base_url", "http://localhost:11434"))
            )

    system_prompt = (
        "You check whether a QUOTE from a sustainability report supports the "
        "CLAIM attributed to it. Judge evidential support only. Preserve "
        "negation, years, numbers, units, and whether the commitment is "
        "planned, ongoing, or achieved. Return one JSON object with exactly "
        'these keys: "support" (one of "supported", "uncertain", '
        '"not_supported"), "confidence" (0 to 1), and "reason" (a short '
        "phrase). Do not add markdown or any other text."
    )
    user_prompt = (
        f"CLAIM:\n{claim}\n\nQUOTE:\n{quote}\n\nReturn the JSON object now."
    )
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={
                "temperature": float(llm_cfg.get("temperature", 0)),
                "num_ctx": int(llm_cfg.get("num_ctx", 4096)),
                "num_predict": 200,
            },
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 - local runtime failure
        return SemanticResult(
            status="not_checked",
            method="llm",
            reason=f"local semantic judge unavailable: {exc}",
        )

    if isinstance(response, dict):
        message = response.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
    else:  # pragma: no cover - newer ollama client response objects
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""

    try:
        from .judge import parse_json_object

        data = parse_json_object(str(content))
        status = str(data.get("support", "uncertain")).strip().lower()
        if status not in {"supported", "uncertain", "not_supported"}:
            status = "uncertain"
        score = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", "") or "").strip()
    except Exception as exc:  # noqa: BLE001 - malformed local output
        return SemanticResult(
            status="not_checked",
            method="llm",
            reason=f"local semantic judge returned unusable output: {exc}",
        )

    return SemanticResult(
        status=status,
        score=max(0.0, min(1.0, score)),
        method="llm",
        reason=reason,
    )


def check_semantic_support(
    claim: str,
    quote: str,
    config: dict[str, Any] | None = None,
    *,
    nli_scorer: Callable[[str, str], float] | None = None,
    llm_client: Any | None = None,
) -> SemanticResult:
    """Check semantic support for one claim/quote pair.

    ``nli_scorer`` and ``llm_client`` are injection points for tests. They are
    not used by the CLI.
    """
    cfg = config or {}
    claim = str(claim or "").strip()
    quote = str(quote or "").strip()
    if not claim or not quote:
        return SemanticResult(
            status="not_checked",
            method="",
            reason="claim or quote is empty",
        )

    guard = consistency_check(claim, quote)
    if guard.status == "uncertain":
        return guard

    verification = get(cfg, "verification", default={}) or {}
    method = str(verification.get("semantic_method", "nli")).lower()

    if method == "llm":
        result = _llm_support(claim, quote, cfg, client=llm_client)
    elif method == "nli":
        try:
            if nli_scorer is not None:
                score = float(nli_scorer(claim, quote))
            else:
                score = _nli_entailment_probability(claim, quote, cfg)
        except ImportError as exc:
            return SemanticResult(
                status="not_checked",
                method="nli",
                reason=(
                    "sentence-transformers is not installed; semantic support "
                    "was not checked"
                ),
                details={"error": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 - missing model, memory, etc.
            LOGGER.warning("NLI semantic check unavailable: %s", exc)
            return SemanticResult(
                status="not_checked",
                method="nli",
                reason=f"NLI model unavailable: {exc}",
            )
        status = _status_from_score(score, cfg)
        return SemanticResult(
            status=status,
            score=round(score, 4),
            method="nli",
            reason=(
                f"NLI entailment probability {score:.2f} "
                f"({'supported' if status == 'supported' else status.replace('_', ' ')})"
            ),
        )
    else:
        return SemanticResult(
            status="not_checked",
            method=method or "unknown",
            reason=f"unknown semantic_method: {method!r}",
        )

    # An explicit LLM decision is authoritative, but the deterministic guard
    # already returned above when it found a mismatch.
    return result


def apply_semantic_support(
    findings: Sequence[VerifiedFinding],
    config: dict[str, Any] | None = None,
    *,
    nli_scorer: Callable[[str, str], float] | None = None,
    llm_client: Any | None = None,
) -> list[VerifiedFinding]:
    """Attach Stage 2 results to verified findings and gate scoring.

    Findings with ``not_supported`` or ``uncertain`` support are marked for
    human review and can no longer be scored. Findings that could not be
    checked remain scorable under Stage 1.
    """
    cfg = config or {}
    for finding in findings:
        if not finding.quote_verified or finding.review_reason is not None:
            continue
        result = check_semantic_support(
            finding.judged.claim,
            finding.judged.supporting_quote,
            cfg,
            nli_scorer=nli_scorer,
            llm_client=llm_client,
        )
        finding.semantic_support = result.status
        finding.semantic_score = result.score
        if result.blocks_scoring and finding.review_reason is None:
            finding.review_reason = (
                "unscored (needs human review): semantic support "
                f"{result.status}: {result.reason or 'no reason recorded'}"
            )
            finding.level = None
            LOGGER.info(
                "Semantic support %s for %s page %s",
                result.status,
                finding.judged.candidate.passage_id,
                finding.source_page,
            )
    return list(findings)
