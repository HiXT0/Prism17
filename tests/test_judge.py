"""Tests for the local Ollama judge's JSON handling and cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism17.config import load_config
from prism17.judge import (
    JudgeOutputError,
    JudgeUnavailableError,
    OllamaJudge,
    RelevanceJudge,
    build_relevance_prompt,
    build_user_prompt,
    parse_json_object,
    validate_judge_data,
    validate_relevance_data,
)
from prism17.models import Candidate


def _candidate() -> Candidate:
    return Candidate(
        passage_id="p0042-001",
        page=42,
        target_id="13.2",
        target_goal=13,
        target_text="Integrate climate change measures into policy.",
        score=0.88,
        passage_text=(
            "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 "
            "against a FY2020 baseline of 100,000 tCO2e."
        ),
        section_heading="Climate",
    )


def _valid_payload() -> dict:
    return {
        "relevant": True,
        "sdg_goal": 13,
        "sdg_target": "13.2",
        "claim": "The company reduced emissions by 42% against a baseline.",
        "supporting_quote": (
            "We reduced Scope 1 and Scope 2 emissions by 42% in FY2024 "
            "against a FY2020 baseline of 100,000 tCO2e."
        ),
        "dimensions": {
            "specificity": True,
            "measurable_indicator": True,
            "deadline": False,
            "baseline": True,
            "reported_progress": True,
        },
        "extracted": {
            "value": "42",
            "unit": "%",
            "year": "2024",
            "status": "achieved",
            "negated": False,
        },
        "confidence": 0.91,
    }


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, **kwargs):  # noqa: ANN003, ANN201 - mimics ollama.Client
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeClient ran out of responses")
        return {"message": {"content": self.responses.pop(0)}}


def _config(tmp_path: Path) -> dict:
    config = load_config()
    config["llm"] = {
        **config["llm"],
        "model": "test-model",
        "cache_dir": str(tmp_path / "llm-cache"),
        "retries": 1,
    }
    return config


def test_parse_json_object_accepts_markdown_fence() -> None:
    raw = 'Here is the result:\n```json\n{"relevant": true}\n```'
    assert parse_json_object(raw) == {"relevant": True}


def test_parse_json_object_rejects_empty_and_invalid() -> None:
    with pytest.raises(JudgeOutputError):
        parse_json_object("")
    with pytest.raises(JudgeOutputError):
        parse_json_object("no object here")
    with pytest.raises(JudgeOutputError):
        parse_json_object("{not valid json}")


def test_build_user_prompt_contains_page_and_target() -> None:
    prompt = build_user_prompt(_candidate())
    assert "page 42" in prompt
    assert "13.2" in prompt
    assert "FY2020 baseline" in prompt


def test_validate_judge_data_normalises_payload() -> None:
    data = validate_judge_data(_valid_payload(), _candidate(), {"13.2"})
    assert data["sdg_goal"] == 13
    assert data["confidence"] == 0.91
    assert data["dimensions"]["baseline"] is True


def test_validate_judge_data_rejects_unknown_target() -> None:
    payload = _valid_payload()
    payload["sdg_target"] = "14.1"
    with pytest.raises(JudgeOutputError):
        validate_judge_data(payload, _candidate(), {"13.2"})


def test_validate_judge_data_rejects_mismatched_candidate_target() -> None:
    payload = _valid_payload()
    payload["sdg_target"] = "13.1"
    with pytest.raises(JudgeOutputError):
        validate_judge_data(payload, _candidate(), {"13.1", "13.2"})


def test_validate_judge_data_rejects_missing_dimension() -> None:
    payload = _valid_payload()
    payload["dimensions"] = {"specificity": True}
    with pytest.raises(JudgeOutputError):
        validate_judge_data(payload, _candidate())


def test_validate_judge_data_rejects_non_numeric_confidence() -> None:
    payload = _valid_payload()
    payload["confidence"] = "high"
    with pytest.raises(JudgeOutputError):
        validate_judge_data(payload, _candidate())


def test_validate_judge_data_clamps_confidence() -> None:
    payload = _valid_payload()
    payload["confidence"] = 4.5
    assert validate_judge_data(payload, _candidate())["confidence"] == 1.0


def test_validate_judge_data_accepts_string_negated_flag() -> None:
    payload = _valid_payload()
    payload["extracted"]["negated"] = "true"
    data = validate_judge_data(payload, _candidate())
    assert data["extracted"]["negated"] is True


def test_validate_judge_data_rejects_invalid_negated_flag() -> None:
    payload = _valid_payload()
    payload["extracted"]["negated"] = "maybe"
    with pytest.raises(JudgeOutputError):
        validate_judge_data(payload, _candidate())


def test_judge_uses_cache_on_second_call(tmp_path: Path) -> None:
    client = FakeClient([json.dumps(_valid_payload())])
    judge = OllamaJudge(_config(tmp_path), client=client)
    candidate = _candidate()

    first = judge.judge(candidate)
    second = judge.judge(candidate)

    assert first.relevant and second.relevant
    assert client.calls == 1


def test_judge_retries_once_after_malformed_output(tmp_path: Path) -> None:
    client = FakeClient(["not json", json.dumps(_valid_payload())])
    judge = OllamaJudge(_config(tmp_path), client=client)

    finding = judge.judge(_candidate())

    assert finding.relevant
    assert client.calls == 2


def test_judge_returns_unscored_after_two_malformed_outputs(tmp_path: Path) -> None:
    client = FakeClient(["not json", "still not json"])
    judge = OllamaJudge(_config(tmp_path), client=client)

    finding = judge.judge(_candidate())

    assert not finding.scored
    assert finding.review_reason is not None
    assert "malformed" in finding.review_reason
    assert client.calls == 2


def test_malformed_result_is_cached(tmp_path: Path) -> None:
    client = FakeClient(["not json", "still not json"])
    judge = OllamaJudge(_config(tmp_path), client=client)
    candidate = _candidate()

    first = judge.judge(candidate)
    second = judge.judge(candidate)

    assert first.review_reason == second.review_reason
    assert client.calls == 2


class FailingClient:
    def chat(self, **kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("local server stopped")


def test_judge_raises_unavailable_when_runtime_fails(tmp_path: Path) -> None:
    judge = OllamaJudge(_config(tmp_path), client=FailingClient())
    with pytest.raises(JudgeUnavailableError):
        judge.judge(_candidate())


def _relevance_payload(relevant: bool = True) -> dict:
    return {
        "relevant": relevant,
        "confidence": 0.8,
        "reason": "The text discusses the target topic.",
    }


def test_build_relevance_prompt_contains_target_and_text() -> None:
    prompt = build_relevance_prompt(
        text="We installed solar panels.",
        target_id="7.2",
        target_goal=7,
        target_text="Increase renewable energy share.",
    )
    assert "7.2" in prompt
    assert "solar panels" in prompt
    assert "renewable energy" in prompt


def test_validate_relevance_data_normalises_and_clamps() -> None:
    data = validate_relevance_data(
        {"relevant": True, "confidence": 4.5, "reason": "  ok  "}
    )
    assert data == {"relevant": True, "confidence": 1.0, "reason": "ok"}


def test_validate_relevance_data_rejects_non_boolean_relevant() -> None:
    with pytest.raises(JudgeOutputError):
        validate_relevance_data({"relevant": "yes", "confidence": 0.5})


def test_relevance_judge_uses_cache_on_second_call(tmp_path: Path) -> None:
    client = FakeClient([json.dumps(_relevance_payload())])
    judge = RelevanceJudge(_config(tmp_path), client=client)

    first = judge.judge(
        "We installed solar panels.",
        target_id="7.2",
        target_goal=7,
        target_text="Increase renewable energy share.",
    )
    second = judge.judge(
        "We installed solar panels.",
        target_id="7.2",
        target_goal=7,
        target_text="Increase renewable energy share.",
    )

    assert first.ok and first.relevant is True
    assert second.ok and second.relevant is True
    assert client.calls == 1


def test_relevance_judge_retries_after_malformed_output(tmp_path: Path) -> None:
    client = FakeClient(["not json", json.dumps(_relevance_payload(False))])
    judge = RelevanceJudge(_config(tmp_path), client=client)

    judgement = judge.judge(
        "Water use in the factory.",
        target_id="6.4",
        target_goal=6,
        target_text="Increase water-use efficiency.",
    )

    assert judgement.ok
    assert judgement.relevant is False
    assert client.calls == 2


def test_relevance_judge_returns_error_after_two_malformed_outputs(
    tmp_path: Path,
) -> None:
    client = FakeClient(["not json", "still not json"])
    judge = RelevanceJudge(_config(tmp_path), client=client)

    judgement = judge.judge(
        "Water use in the factory.",
        target_id="6.4",
        target_goal=6,
        target_text="Increase water-use efficiency.",
    )

    assert not judgement.ok
    assert judgement.relevant is None
    assert judgement.error is not None
    assert client.calls == 2


def test_relevance_judge_caches_errors(tmp_path: Path) -> None:
    client = FakeClient(["not json", "still not json"])
    judge = RelevanceJudge(_config(tmp_path), client=client)

    first = judge.judge(
        "Water use in the factory.",
        target_id="6.4",
        target_goal=6,
        target_text="Increase water-use efficiency.",
    )
    second = judge.judge(
        "Water use in the factory.",
        target_id="6.4",
        target_goal=6,
        target_text="Increase water-use efficiency.",
    )

    assert first.error == second.error
    assert client.calls == 2
