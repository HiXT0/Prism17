"""Tests for deterministic evaluation metrics and provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism17.evaluation import (
    append_jsonl,
    binary_metrics,
    categorical_agreement,
    cohen_kappa,
    default_thresholds,
    evaluate_threshold,
    file_info,
    macro_f1_over_groups,
    per_group_metrics,
    quadratic_weighted_kappa,
    rubric_agreement,
    select_threshold,
    threshold_grid,
)


def _row(
    sdg: int,
    label: int,
    score: float,
    *,
    status: str = "ok",
    relevant: bool | None = None,
) -> dict:
    return {
        "sdg": sdg,
        "label": label,
        "best_score": score,
        "best_target_id": "13.2" if score > 0 else None,
        "judge_status": status,
        "judge_relevant": relevant,
    }


def test_binary_metrics_counts_and_positive_class_scores() -> None:
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["macro_f1"] == 0.5


def test_binary_metrics_all_positive_prediction_is_not_rewarded_by_macro_f1() -> None:
    metrics = binary_metrics([1, 1, 1, 0], [1, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
    # Positive F1 is 0.857; negative F1 is 0 because the only negative row
    # was predicted positive. The two-class average must not reach 1.0.
    assert metrics["macro_f1"] == pytest.approx(0.428571, abs=1e-6)


def test_binary_metrics_handles_empty_and_zero_division() -> None:
    metrics = binary_metrics([], [])
    assert metrics["n"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_binary_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        binary_metrics([1], [1, 0])


def test_per_group_metrics_sorts_numeric_groups() -> None:
    grouped = per_group_metrics(
        [12, 3, 12, 3],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
    )
    assert list(grouped) == ["3", "12"]
    assert grouped["3"]["n"] == 2
    assert grouped["12"]["n"] == 2


def test_macro_f1_skips_groups_with_only_one_class() -> None:
    grouped = {
        "3": binary_metrics([1, 0], [1, 0]),
        "12": binary_metrics([1, 1], [1, 1]),
    }
    value, count = macro_f1_over_groups(grouped)
    assert count == 1
    assert value == 1.0


def test_default_thresholds_is_inclusive_and_rounded() -> None:
    assert default_thresholds(0.0, 0.05, 0.02) == [0.0, 0.02, 0.04]
    assert default_thresholds(0.3, 0.3, 0.01) == [0.3]
    with pytest.raises(ValueError):
        default_thresholds(0.0, 0.1, 0.0)


def test_evaluate_threshold_separates_retrieval_and_judge_stages() -> None:
    rows = [
        _row(13, 1, 0.9, relevant=True),
        _row(13, 0, 0.8, relevant=False),
        _row(13, 1, 0.2, relevant=True),
        _row(13, 0, 0.1, status="error"),
        _row(12, 1, 0.7, status="error"),
        _row(12, 0, 0.6, relevant=False),
    ]
    result = evaluate_threshold(rows, 0.4)

    retrieval = result["retrieval_only"]
    assert retrieval["overall"]["tp"] == 2
    assert retrieval["overall"]["fp"] == 2
    assert retrieval["overall"]["fn"] == 1
    assert retrieval["overall"]["tn"] == 1

    judge = result["retrieve_judge"]
    assert judge["overall"]["tp"] == 1
    assert judge["overall"]["fp"] == 0
    assert judge["overall"]["fn"] == 1
    assert judge["overall"]["tn"] == 3
    assert judge["judge_excluded"] == 1
    assert judge["judge_coverage"] == pytest.approx(5 / 6)


def test_evaluate_threshold_never_predicts_positive_without_a_candidate() -> None:
    rows = [_row(13, 1, 0.0, status="no_same_goal_candidate")]
    result = evaluate_threshold(rows, 0.0)
    assert result["retrieval_only"]["overall"]["tp"] == 0
    assert result["retrieval_only"]["overall"]["fn"] == 1


def test_evaluate_threshold_excludes_unjuged_surfaced_rows() -> None:
    rows = [_row(13, 1, 0.9, status="not_run")]
    result = evaluate_threshold(rows, 0.4)
    assert result["retrieval_only"]["overall"]["n"] == 1
    assert result["retrieve_judge"]["overall"]["n"] == 0
    assert result["retrieve_judge"]["judge_excluded"] == 1


def test_threshold_grid_evaluates_each_threshold() -> None:
    rows = [_row(13, 1, 0.5, relevant=True)]
    grid = threshold_grid(rows, [0.4, 0.6])
    assert [entry["threshold"] for entry in grid] == [0.4, 0.6]
    assert grid[0]["retrieval_only"]["overall"]["tp"] == 1
    assert grid[1]["retrieval_only"]["overall"]["fn"] == 1


def test_select_threshold_uses_macro_f1_then_accuracy_then_higher() -> None:
    grid = [
        {
            "threshold": 0.3,
            "retrieval_only": {
                "overall": {"accuracy": 0.9, "f1": 0.9},
                "macro_f1": 0.8,
                "macro_f1_groups": 2,
            },
        },
        {
            "threshold": 0.4,
            "retrieval_only": {
                "overall": {"accuracy": 0.7, "f1": 0.7},
                "macro_f1": 0.9,
                "macro_f1_groups": 2,
            },
        },
        {
            "threshold": 0.5,
            "retrieval_only": {
                "overall": {"accuracy": 0.7, "f1": 0.7},
                "macro_f1": 0.9,
                "macro_f1_groups": 2,
            },
        },
    ]
    selected = select_threshold(grid)
    assert selected is not None
    assert selected["threshold"] == 0.5


def test_select_threshold_falls_back_to_overall_f1_without_two_class_groups() -> None:
    grid = [
        {
            "threshold": 0.3,
            "retrieval_only": {
                "overall": {"accuracy": 0.5, "f1": 0.4},
                "macro_f1": 0.0,
                "macro_f1_groups": 0,
            },
        },
        {
            "threshold": 0.4,
            "retrieval_only": {
                "overall": {"accuracy": 0.5, "f1": 0.6},
                "macro_f1": 0.0,
                "macro_f1_groups": 0,
            },
        },
    ]
    selected = select_threshold(grid)
    assert selected is not None
    assert selected["threshold"] == 0.4


def test_file_info_and_append_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("hello", encoding="utf-8")
    info = file_info(source)
    assert info["bytes"] == 5
    assert len(info["sha256"]) == 64
    assert len(info["md5"]) == 32

    log = tmp_path / "runs.jsonl"
    append_jsonl(log, {"run": 1})
    append_jsonl(log, {"run": 2})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run"] for line in lines] == [1, 2]


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_and_chance_agreement() -> None:
    perfect = cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4], labels=[1, 2, 3, 4])
    assert perfect["observed_agreement"] == 1.0
    assert perfect["expected_agreement"] == 0.25
    assert perfect["kappa"] == 1.0

    chance = cohen_kappa([1, 1, 2, 2], [1, 2, 1, 2], labels=[1, 2])
    assert chance["observed_agreement"] == 0.5
    assert chance["expected_agreement"] == 0.5
    assert chance["kappa"] == 0.0


def test_cohen_kappa_handles_empty_and_single_label_input() -> None:
    empty = cohen_kappa([], [])
    assert empty["n"] == 0
    assert empty["kappa"] == 0.0

    single = cohen_kappa([1, 1], [1, 1], labels=[1])
    assert single["expected_agreement"] == 1.0
    assert single["kappa"] == 1.0


def test_cohen_kappa_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])


def test_quadratic_weighted_kappa_rewards_near_misses() -> None:
    # One two-level swap, two exact matches, one one-level miss.
    result = quadratic_weighted_kappa(
        [1, 1, 2, 2], [1, 2, 2, 2], labels=[1, 2]
    )
    assert result["observed_agreement"] == 0.75
    assert result["expected_agreement"] == 0.5
    assert result["kappa"] == 0.5


def test_quadratic_weighted_kappa_is_one_for_identical_ratings() -> None:
    result = quadratic_weighted_kappa(
        [1, 2, 3, 4], [1, 2, 3, 4], labels=[1, 2, 3, 4]
    )
    assert result["kappa"] == 1.0


def test_quadratic_weighted_kappa_handles_empty_input() -> None:
    result = quadratic_weighted_kappa([], [])
    assert result["n"] == 0
    assert result["kappa"] == 0.0


def test_rubric_agreement_reports_exact_within_one_and_kappa() -> None:
    result = rubric_agreement([1, 1, 2, 2], [1, 2, 2, 2])
    assert result["n"] == 4
    assert result["exact_agreement"] == 0.75
    assert result["within_one_agreement"] == 1.0
    assert result["mean_absolute_difference"] == 0.25
    assert result["cohen_kappa"] == 0.5
    assert result["quadratic_weighted_kappa"] == 0.5
    assert result["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_rubric_agreement_handles_empty_input() -> None:
    result = rubric_agreement([], [])
    assert result["n"] == 0
    assert result["exact_agreement"] == 0.0
    assert result["within_one_agreement"] == 0.0
    assert result["mean_absolute_difference"] == 0.0
    assert result["confusion_matrix"] == [[0] * 4 for _ in range(4)]


def test_categorical_agreement_counts_exact_matches_and_kappa() -> None:
    labels = ("supported", "uncertain", "not_supported")
    result = categorical_agreement(
        ["supported", "supported", "uncertain", "not_supported"],
        ["supported", "uncertain", "uncertain", "not_supported"],
        labels=labels,
    )
    assert result["n"] == 4
    assert result["observed_agreement"] == 0.75
    assert result["cohen_kappa"] == 0.636364
    assert result["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    empty = categorical_agreement([], [], labels=labels)
    assert empty["n"] == 0
    assert empty["observed_agreement"] == 0.0
    assert empty["cohen_kappa"] == 0.0
