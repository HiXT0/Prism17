"""Tests for Stage 2 semantic support checking."""

from __future__ import annotations

import json

from prism17 import semantic
from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.semantic import (
    SemanticResult,
    apply_semantic_support,
    check_semantic_support,
    consistency_check,
)

CLAIM = "The company reduced emissions by 42 percent."
QUOTE = "We reduced emissions by 42 percent against the previous year."


def _config(method: str = "nli") -> dict:
    return {
        "verification": {
            "semantic_method": method,
            "nli_supported_threshold": 0.6,
            "nli_uncertain_threshold": 0.35,
        },
        "llm": {"model": "test-model"},
    }


def _finding(
    *,
    quote_verified: bool = True,
    level: int | None = 3,
    review_reason: str | None = None,
    claim: str = CLAIM,
    quote: str = QUOTE,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id="p0049-001",
        page=49,
        target_id="12.5",
        target_goal=12,
        target_text="Substantially reduce waste generation through prevention.",
        score=0.8,
        passage_text=quote,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=True,
        sdg_goal=12,
        sdg_target="12.5",
        claim=claim,
        supporting_quote=quote,
        dimensions={"specificity": True},
        extracted={"value": "42", "unit": "%", "year": "2024"},
        confidence=0.9,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=49,
        level=level if quote_verified else None,
        review_reason=review_reason,
    )


class _FakeClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error

    def chat(self, **_: object) -> dict:
        if self.error is not None:
            raise self.error
        return {"message": {"content": self.content}}


# ---------------------------------------------------------------------------
# Deterministic consistency guard
# ---------------------------------------------------------------------------


def test_consistency_check_accepts_matching_claim_and_quote() -> None:
    result = consistency_check(CLAIM, QUOTE)
    assert result.status == "supported"
    assert result.method == "consistency"
    assert result.blocks_scoring is False


def test_consistency_check_flags_year_missing_from_quote() -> None:
    result = consistency_check(
        "Emissions fell 42 percent in 2024.",
        "Emissions fell 42 percent.",
    )
    assert result.status == "uncertain"
    assert "2024" in result.reason
    assert result.details["missing_years"] == [2024]


def test_consistency_check_flags_number_missing_from_quote() -> None:
    result = consistency_check(
        "The company reduced emissions by 42 percent.",
        "The company reduced emissions by 40 percent.",
    )
    assert result.status == "uncertain"
    assert result.details["missing_numbers"] == [42.0]


def test_consistency_check_flags_different_units() -> None:
    result = consistency_check("We used 10 tCO2e.", "We used 10 MWh.")
    assert result.status == "uncertain"
    assert result.details["claim_units"] == ["tco2e"]
    assert result.details["quote_units"] == ["mwh"]


def test_consistency_check_recognises_percent_as_a_unit() -> None:
    result = consistency_check(
        "We reduced emissions by 42%.", "We reduced emissions by 42 tCO2e."
    )
    assert result.status == "uncertain"
    assert result.details["claim_units"] == ["%"]
    assert result.details["quote_units"] == ["tco2e"]


def test_consistency_check_flags_negation_mismatch() -> None:
    result = consistency_check(
        "The company did not reduce emissions.",
        "The company reduced emissions.",
    )
    assert result.status == "uncertain"
    assert result.details == {"claim_negated": True, "quote_negated": False}


def test_consistency_check_flags_commitment_status_mismatch() -> None:
    result = consistency_check(
        "We achieved a 20 percent reduction.",
        "We plan to reduce by 20 percent.",
    )
    assert result.status == "uncertain"
    assert result.details == {"claim_status": "achieved", "quote_status": "planned"}


def test_consistency_check_ignores_calendar_years_as_measured_values() -> None:
    # A year token must not be compared as if it were a measured value.
    result = consistency_check("In 2024 we reduced waste.", "In 2024 waste fell.")
    assert result.status == "supported"


# ---------------------------------------------------------------------------
# NLI path
# ---------------------------------------------------------------------------


def test_nli_score_above_threshold_is_supported() -> None:
    result = check_semantic_support(
        CLAIM, QUOTE, _config(), nli_scorer=lambda claim, quote: 0.91
    )
    assert result.status == "supported"
    assert result.method == "nli"
    assert result.score == 0.91


def test_nli_score_between_thresholds_is_uncertain() -> None:
    result = check_semantic_support(
        CLAIM, QUOTE, _config(), nli_scorer=lambda claim, quote: 0.45
    )
    assert result.status == "uncertain"


def test_nli_score_below_uncertain_threshold_is_not_supported() -> None:
    result = check_semantic_support(
        CLAIM, QUOTE, _config(), nli_scorer=lambda claim, quote: 0.10
    )
    assert result.status == "not_supported"


def test_nli_missing_dependency_leaves_finding_not_checked(monkeypatch) -> None:
    def _raise_import(claim: str, quote: str, config: dict) -> float:
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(semantic, "_nli_entailment_probability", _raise_import)
    result = check_semantic_support(CLAIM, QUOTE, _config())
    assert result.status == "not_checked"
    assert result.blocks_scoring is False
    assert "not installed" in result.reason


def test_nli_runtime_failure_leaves_finding_not_checked(monkeypatch) -> None:
    def _raise_runtime(claim: str, quote: str, config: dict) -> float:
        raise RuntimeError("model could not be loaded")

    monkeypatch.setattr(semantic, "_nli_entailment_probability", _raise_runtime)
    result = check_semantic_support(CLAIM, QUOTE, _config())
    assert result.status == "not_checked"
    assert "unavailable" in result.reason


# ---------------------------------------------------------------------------
# Local LLM path
# ---------------------------------------------------------------------------


def test_llm_support_uses_structured_local_response() -> None:
    client = _FakeClient(
        json.dumps(
            {"support": "supported", "confidence": 0.88, "reason": "same claim"}
        )
    )
    result = check_semantic_support(
        CLAIM, QUOTE, _config("llm"), llm_client=client
    )
    assert result.status == "supported"
    assert result.method == "llm"
    assert result.score == 0.88


def test_llm_support_coerces_unknown_status_to_uncertain() -> None:
    client = _FakeClient(
        json.dumps({"support": "maybe", "confidence": 0.5, "reason": "unclear"})
    )
    result = check_semantic_support(
        CLAIM, QUOTE, _config("llm"), llm_client=client
    )
    assert result.status == "uncertain"


def test_llm_malformed_output_is_not_checked() -> None:
    client = _FakeClient("this is not json")
    result = check_semantic_support(
        CLAIM, QUOTE, _config("llm"), llm_client=client
    )
    assert result.status == "not_checked"
    assert "unusable output" in result.reason


def test_llm_client_failure_is_not_checked() -> None:
    client = _FakeClient(error=RuntimeError("ollama is not running"))
    result = check_semantic_support(
        CLAIM, QUOTE, _config("llm"), llm_client=client
    )
    assert result.status == "not_checked"
    assert "unavailable" in result.reason


def test_unknown_semantic_method_is_not_checked() -> None:
    result = check_semantic_support(CLAIM, QUOTE, _config("magic"))
    assert result.status == "not_checked"
    assert "unknown semantic_method" in result.reason


def test_empty_claim_or_quote_is_not_checked() -> None:
    assert check_semantic_support("", QUOTE, _config()).status == "not_checked"
    assert check_semantic_support(CLAIM, "  ", _config()).status == "not_checked"


def test_consistency_guard_short_circuits_the_model() -> None:
    called = {"count": 0}

    def _scorer(claim: str, quote: str) -> float:
        called["count"] += 1
        return 0.99

    result = check_semantic_support(
        "We reduced emissions by 42 percent in 2024.",
        "We reduced emissions by 40 percent in 2024.",
        _config(),
        nli_scorer=_scorer,
    )
    assert result.status == "uncertain"
    assert result.method == "consistency"
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# Applying results to findings
# ---------------------------------------------------------------------------


def test_apply_semantic_support_vetoes_low_support_and_clears_level() -> None:
    finding = _finding()
    apply_semantic_support(
        [finding], _config(), nli_scorer=lambda claim, quote: 0.05
    )
    assert finding.semantic_support == "not_supported"
    assert finding.level is None
    assert finding.is_scored is False
    assert finding.review_reason is not None
    assert "semantic support not_supported" in finding.review_reason


def test_apply_semantic_support_vetoes_uncertain_support() -> None:
    finding = _finding()
    apply_semantic_support(
        [finding], _config(), nli_scorer=lambda claim, quote: 0.45
    )
    assert finding.semantic_support == "uncertain"
    assert finding.is_scored is False
    assert finding.level is None


def test_apply_semantic_support_keeps_supported_finding_scored() -> None:
    finding = _finding()
    apply_semantic_support(
        [finding], _config(), nli_scorer=lambda claim, quote: 0.95
    )
    assert finding.semantic_support == "supported"
    assert finding.is_scored is True
    assert finding.level == 3


def test_apply_semantic_support_keeps_finding_scored_when_not_checked(
    monkeypatch,
) -> None:
    def _raise_import(claim: str, quote: str, config: dict) -> float:
        raise ImportError("optional dependency missing")

    monkeypatch.setattr(semantic, "_nli_entailment_probability", _raise_import)
    finding = _finding()
    apply_semantic_support([finding], _config())
    assert finding.semantic_support == "not_checked"
    assert finding.is_scored is True


def test_apply_semantic_support_skips_unverified_and_already_reviewed() -> None:
    unverified = _finding(quote_verified=False)
    reviewed = _finding(review_reason="already queued for review")
    apply_semantic_support(
        [unverified, reviewed], _config(), nli_scorer=lambda claim, quote: 0.01
    )
    assert unverified.semantic_support == "not_checked"
    assert reviewed.semantic_support == "not_checked"
    assert reviewed.review_reason == "already queued for review"


def test_semantic_result_blocks_scoring_for_blocking_statuses() -> None:
    assert SemanticResult(status="supported").blocks_scoring is False
    assert SemanticResult(status="not_checked").blocks_scoring is False
    assert SemanticResult(status="uncertain").blocks_scoring is True
    assert SemanticResult(status="not_supported").blocks_scoring is True
