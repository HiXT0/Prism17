"""Tests for Stage 2 semantic support verification."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from prism17.judge import SupportJudgement
from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.semantic import (
    NOT_CHECKED,
    NOT_SUPPORTED,
    SUPPORTED,
    UNCERTAIN,
    deterministic_check,
    llm_check,
    nli_check,
    verify_semantics,
)


def _config(**verification) -> dict:
    return {
        "verification": {
            "semantic_enabled": True,
            "semantic_method": "nli",
            "nli_supported_threshold": 0.60,
            "nli_uncertain_threshold": 0.35,
            **verification,
        }
    }


def _finding(
    claim: str = "We reduced emissions by 40%.",
    quote: str = "We reduced emissions by 40% against a 2020 baseline.",
    *,
    relevant: bool = True,
    quote_verified: bool = True,
    level: int | None = 3,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id="p0001-001",
        page=1,
        target_id="13.2",
        target_goal=13,
        target_text="Integrate climate change measures into policy.",
        score=0.9,
        passage_text=quote,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=13,
        sdg_target="13.2",
        claim=claim,
        supporting_quote=quote,
        dimensions={
            "specificity": True,
            "measurable_indicator": True,
            "deadline": True,
            "baseline": False,
            "reported_progress": False,
        },
        confidence=0.9,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=1,
        level=level,
    )


class FakeNliModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.config = SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        self.calls = 0

    def predict(self, pairs, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls += 1
        return np.array([self.probabilities for _ in pairs], dtype="float32")


def test_deterministic_check_rejects_no_content_overlap() -> None:
    check = deterministic_check("We planted trees.", "Water use fell by 10%.")
    assert check is not None
    assert check.support == NOT_SUPPORTED


def test_deterministic_check_allows_overlap_for_nli() -> None:
    assert (
        deterministic_check(
            "We reduced emissions by 40%.",
            "Emissions fell 40% against a baseline.",
        )
        is None
    )


def test_deterministic_check_rejects_empty_claim_or_quote() -> None:
    assert deterministic_check("", "quote").support == NOT_SUPPORTED
    assert deterministic_check("claim", "").support == NOT_SUPPORTED


def test_nli_check_maps_high_entailment_to_supported() -> None:
    model = FakeNliModel([0.1, 0.8, 0.1])
    check = nli_check(
        "We reduced emissions by 40%.",
        "We reduced emissions by 40% against a 2020 baseline.",
        _config(),
        model=model,
    )
    assert check.support == SUPPORTED
    assert check.score == 0.8
    assert model.calls == 1


def test_nli_check_maps_mid_entailment_to_uncertain() -> None:
    check = nli_check(
        "We reduced emissions by 40%.",
        "We reduced emissions by 40% against a 2020 baseline.",
        _config(),
        model=FakeNliModel([0.2, 0.45, 0.35]),
    )
    assert check.support == UNCERTAIN


def test_nli_check_maps_low_entailment_to_not_supported() -> None:
    check = nli_check(
        "We reduced emissions by 40%.",
        "We reduced emissions by 40% against a 2020 baseline.",
        _config(),
        model=FakeNliModel([0.6, 0.2, 0.2]),
    )
    assert check.support == NOT_SUPPORTED


def test_nli_check_falls_back_to_uncertain_when_model_fails() -> None:
    class BrokenModel:
        config = SimpleNamespace(id2label={1: "entailment"})

        def predict(self, pairs, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("model unavailable")

    check = nli_check(
        "We reduced emissions by 40%.",
        "We reduced emissions by 40% against a 2020 baseline.",
        _config(),
        model=BrokenModel(),
    )
    assert check.support == UNCERTAIN
    assert "unavailable" in (check.reason or "")


def test_llm_check_maps_judge_labels() -> None:
    class FakeJudge:
        def judge(self, claim: str, quote: str) -> SupportJudgement:
            return SupportJudgement(
                supported="yes",
                confidence=0.9,
                reason="faithful restatement",
                status="ok",
            )

    check = llm_check(
        "We reduced emissions by 40%.",
        "We reduced emissions by 40% against a 2020 baseline.",
        _config(semantic_method="llm"),
        judge=FakeJudge(),
    )
    assert check.support == SUPPORTED
    assert check.method == "llm"


def test_verify_semantics_disabled_marks_not_checked() -> None:
    finding = _finding()
    verify_semantics([finding], _config(), enabled=False)
    assert finding.semantic_support == NOT_CHECKED
    assert finding.is_scored


def test_verify_semantics_blocks_unsupported_finding() -> None:
    finding = _finding()
    verify_semantics(
        [finding],
        _config(),
        model=FakeNliModel([0.8, 0.1, 0.1]),
    )
    assert finding.semantic_support == NOT_SUPPORTED
    assert finding.review_reason is not None
    assert not finding.is_scored


def test_verify_semantics_keeps_supported_finding_scored() -> None:
    finding = _finding()
    verify_semantics(
        [finding],
        _config(),
        model=FakeNliModel([0.1, 0.8, 0.1]),
    )
    assert finding.semantic_support == SUPPORTED
    assert finding.review_reason is None
    assert finding.is_scored


def test_verify_semantics_skips_findings_that_failed_stage_one() -> None:
    finding = _finding(quote_verified=False)
    finding.review_reason = "quote could not be verified"
    verify_semantics(
        [finding],
        _config(),
        model=FakeNliModel([0.1, 0.8, 0.1]),
    )
    assert finding.semantic_support == NOT_CHECKED
    assert finding.review_reason == "quote could not be verified"
