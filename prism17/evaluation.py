"""Reusable evaluation helpers.

This module contains deterministic metric code and provenance helpers only. It
does not call a model. Evaluation scripts use it to keep their numbers
comparable across runs and to record exactly which dataset and files produced
each result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Return a hexadecimal digest for a file without loading it all at once."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: str | Path) -> dict[str, Any]:
    """Return size and checksums for a file used by an evaluation run."""
    resolved = Path(path)
    return {
        "path": str(resolved),
        "name": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": file_digest(resolved, "sha256"),
        "md5": file_digest(resolved, "md5"),
    }


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    """Return installed versions for the requested distributions."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict[str, Any]:
    """Return interpreter and package details for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "sentence-transformers",
                "faiss-cpu",
                "ollama",
                "torch",
                "transformers",
                "rapidfuzz",
            ]
        ),
    }


def build_provenance(
    *,
    dataset: dict[str, Any],
    files: dict[str, Any],
    config: dict[str, Any],
    sample: dict[str, Any],
    notes: Sequence[str] = (),
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance record written with every evaluation result.

    ``dataset`` should include at least a name, version, and source DOI.
    ``files`` should map a role (for example ``"labels"``) to :func:`file_info`
    output or an equivalent dictionary.
    """
    return {
        "created_at_utc": utc_now_iso(),
        "dataset": dict(dataset),
        "files": dict(files),
        "config": dict(config),
        "sample": dict(sample),
        "environment": environment or environment_info(),
        "notes": list(notes),
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one JSON object to a JSONL run log, creating parents if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Return positive-class and two-class metrics for binary predictions.

    ``precision``, ``recall``, and ``f1`` refer to the positive (relevant)
    class. ``macro_f1`` averages the positive and negative class F1 scores, so
    an all-positive prediction on imbalanced data is not rewarded. Division by
    zero yields 0.0 rather than an error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = fp = tn = fn = 0
    for truth, prediction in zip(y_true, y_pred):
        truth_value = int(bool(truth))
        prediction_value = int(bool(prediction))
        if truth_value and prediction_value:
            tp += 1
        elif not truth_value and prediction_value:
            fp += 1
        elif not truth_value and not prediction_value:
            tn += 1
        else:
            fn += 1

    positive = _prf(tp, fp, fn)
    negative = _prf(tn, fn, fp)
    total = tp + fp + tn + fn
    accuracy = _safe_ratio(tp + tn, total)
    balanced_accuracy = (positive["recall"] + negative["recall"]) / 2

    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1"],
        "negative_precision": negative["precision"],
        "negative_recall": negative["recall"],
        "negative_f1": negative["f1"],
        "macro_f1": round((positive["f1"] + negative["f1"]) / 2, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
    }


def per_group_metrics(
    groups: Sequence[Any],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Compute :func:`binary_metrics` separately for each group label."""
    if not (len(groups) == len(y_true) == len(y_pred)):
        raise ValueError("groups, y_true, and y_pred must have the same length")

    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for group, truth, prediction in zip(groups, y_true, y_pred):
        key = str(group)
        truths, predictions = grouped.setdefault(key, ([], []))
        truths.append(int(bool(truth)))
        predictions.append(int(bool(prediction)))

    return {
        key: binary_metrics(truths, predictions)
        for key, (truths, predictions) in sorted(
            grouped.items(), key=lambda item: _group_sort_key(item[0])
        )
    }


def _group_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def macro_f1_over_groups(
    grouped_metrics: dict[str, dict[str, Any]],
    *,
    require_both_classes: bool = True,
) -> tuple[float, int]:
    """Average per-group macro-F1 over groups that support the estimate.

    Groups with only positives or only negatives cannot produce a meaningful
    two-class macro-F1, so they are skipped by default. Returns the average and
    the number of groups included.
    """
    values: list[float] = []
    for metrics in grouped_metrics.values():
        positives = int(metrics["tp"]) + int(metrics["fn"])
        negatives = int(metrics["tn"]) + int(metrics["fp"])
        if require_both_classes and (positives == 0 or negatives == 0):
            continue
        values.append(float(metrics["macro_f1"]))
    if not values:
        return 0.0, 0
    return round(sum(values) / len(values), 6), len(values)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def _validate_ratings(
    first: Sequence[Any], second: Sequence[Any]
) -> None:
    if len(first) != len(second):
        raise ValueError("rating sequences must have the same length")


def _rating_labels(
    first: Sequence[Any], second: Sequence[Any], labels: Sequence[Any] | None
) -> list[Any]:
    if labels is not None:
        return list(labels)
    return sorted(set(first) | set(second))


def _kappa_from_agreement(observed: float, expected: float) -> float:
    if expected >= 1.0:
        # Perfect expected agreement means the ratings carry no information.
        # Matching that agreement is conventionally reported as 1.0.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _confusion_matrix(
    first: Sequence[Any],
    second: Sequence[Any],
    labels: Sequence[Any],
) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for left, right in zip(first, second):
        if left not in index or right not in index:
            continue
        matrix[index[left]][index[right]] += 1
    return matrix


def cohen_kappa(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return unweighted Cohen's kappa for two rating sequences.

    ``labels`` fixes the rating scale. When omitted, the sorted union of the
    observed ratings is used. Empty input and a degenerate expected agreement
    return a kappa of 0.0 rather than dividing by zero.
    """
    _validate_ratings(first, second)
    scale = _rating_labels(first, second, labels)
    total = len(first)
    if total == 0 or not scale:
        return {
            "n": 0,
            "labels": scale,
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "kappa": 0.0,
        }

    matrix = _confusion_matrix(first, second, scale)
    observed = sum(matrix[index][index] for index in range(len(scale))) / total
    first_totals = [sum(row) for row in matrix]
    second_totals = [
        sum(matrix[row][column] for row in range(len(scale)))
        for column in range(len(scale))
    ]
    expected = sum(
        first_totals[index] * second_totals[index]
        for index in range(len(scale))
    ) / (total * total)

    return {
        "n": total,
        "labels": scale,
        "observed_agreement": round(observed, 6),
        "expected_agreement": round(expected, 6),
        "kappa": round(_kappa_from_agreement(observed, expected), 6),
    }


def quadratic_weighted_kappa(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return quadratic weighted kappa for ordered ratings.

    The labels are treated as an ordered scale. A one-level difference is
    penalised less than a two-level difference, which suits rubric levels better
    than unweighted kappa.
    """
    _validate_ratings(first, second)
    scale = _rating_labels(first, second, labels)
    total = len(first)
    if total == 0 or len(scale) < 2:
        return {
            "n": total,
            "labels": scale,
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "kappa": 0.0,
        }

    matrix = _confusion_matrix(first, second, scale)
    size = len(scale)
    denominator_scale = float((size - 1) ** 2)
    weights = [
        [
            ((left - right) ** 2) / denominator_scale
            for right in range(size)
        ]
        for left in range(size)
    ]

    first_totals = [sum(row) / total for row in matrix]
    second_totals = [
        sum(matrix[row][column] for row in range(size)) / total
        for column in range(size)
    ]
    observed_disagreement = sum(
        weights[left][right] * matrix[left][right] / total
        for left in range(size)
        for right in range(size)
    )
    expected_disagreement = sum(
        weights[left][right] * first_totals[left] * second_totals[right]
        for left in range(size)
        for right in range(size)
    )
    kappa = (
        1.0 - observed_disagreement / expected_disagreement
        if expected_disagreement > 0
        else (1.0 if observed_disagreement == 0 else 0.0)
    )

    return {
        "n": total,
        "labels": scale,
        "observed_agreement": round(1.0 - observed_disagreement, 6),
        "expected_agreement": round(1.0 - expected_disagreement, 6),
        "kappa": round(kappa, 6),
    }


def rubric_agreement(
    reviewer_levels: Sequence[Any],
    model_levels: Sequence[Any],
    *,
    labels: Sequence[Any] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Summarise agreement between reviewer and model rubric levels.

    Returns exact agreement, agreement within one level, mean absolute
    difference, unweighted and quadratic weighted kappa, and the confusion
    matrix (rows are the first sequence, columns the second).
    """
    _validate_ratings(reviewer_levels, model_levels)
    total = len(reviewer_levels)
    scale = list(labels)
    matrix = _confusion_matrix(reviewer_levels, model_levels, scale)

    if total == 0:
        exact = within_one = mean_difference = 0.0
    else:
        differences = [
            abs(float(left) - float(right))
            for left, right in zip(reviewer_levels, model_levels)
        ]
        exact = sum(1 for value in differences if value == 0) / total
        within_one = sum(1 for value in differences if value <= 1) / total
        mean_difference = sum(differences) / total

    return {
        "n": total,
        "labels": scale,
        "exact_agreement": round(exact, 6),
        "within_one_agreement": round(within_one, 6),
        "mean_absolute_difference": round(mean_difference, 6),
        "cohen_kappa": cohen_kappa(
            reviewer_levels, model_levels, labels=scale
        )["kappa"],
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            reviewer_levels, model_levels, labels=scale
        )["kappa"],
        "confusion_matrix": matrix,
    }


def categorical_agreement(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    labels: Sequence[Any],
) -> dict[str, Any]:
    """Summarise agreement on an unordered categorical scale.

    Used for semantic-support labels such as ``supported``, ``uncertain``,
    and ``not_supported``. Rows are the first sequence and columns the second.
    """
    _validate_ratings(first, second)
    scale = list(labels)
    total = len(first)
    matrix = _confusion_matrix(first, second, scale)
    agreed = (
        sum(matrix[index][index] for index in range(len(scale)))
        if total
        else 0
    )
    return {
        "n": total,
        "labels": scale,
        "observed_agreement": round(agreed / total, 6) if total else 0.0,
        "cohen_kappa": cohen_kappa(first, second, labels=scale)["kappa"],
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------


def default_thresholds(
    start: float = 0.0,
    stop: float = 0.8,
    step: float = 0.01,
) -> list[float]:
    """Return an inclusive, rounded threshold grid."""
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    values: list[float] = []
    current = round(float(start), 6)
    stop_value = round(float(stop), 6)
    # Round each step to avoid floating-point drift at the upper bound.
    while current <= stop_value + 1e-9:
        values.append(current)
        current = round(current + step, 6)
    return values


def evaluate_threshold(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    group_key: str = "sdg",
    label_key: str = "label",
    score_key: str = "best_score",
    judge_status_key: str = "judge_status",
    judge_relevant_key: str = "judge_relevant",
    has_candidate_key: str = "best_target_id",
) -> dict[str, Any]:
    """Evaluate retrieval-only and retrieve-then-judge predictions.

    Each row must provide a binary ``label``, a ``best_score`` for the
    suggested goal, and the judge fields. Retrieval-only predicts positive when
    a candidate exists and ``best_score >= threshold``. A row with no same-goal
    candidate is always negative, even at threshold 0.0. Retrieve-then-judge
    additionally requires the judge to mark the surfaced candidate relevant.

    Rows whose top candidate clears the threshold but whose judge call failed
    are excluded from the retrieve-then-judge metrics and counted in
    ``judge_excluded``. Rows below the threshold need no judge decision.
    """
    retrieval_truth: list[int] = []
    retrieval_pred: list[int] = []
    retrieval_groups: list[Any] = []

    judge_truth: list[int] = []
    judge_pred: list[int] = []
    judge_groups: list[Any] = []
    judge_excluded = 0

    for row in rows:
        truth = int(bool(row[label_key]))
        score = float(row.get(score_key) or 0.0)
        group = row.get(group_key)
        if has_candidate_key in row:
            has_candidate = row.get(has_candidate_key) is not None
        else:
            # Older callers may not record the candidate id. A positive score
            # is the best available signal that retrieval returned something.
            has_candidate = score > 0.0
        surfaced = has_candidate and score >= threshold

        retrieval_truth.append(truth)
        retrieval_pred.append(1 if surfaced else 0)
        retrieval_groups.append(group)

        if not surfaced:
            judge_truth.append(truth)
            judge_pred.append(0)
            judge_groups.append(group)
            continue

        status = str(row.get(judge_status_key) or "not_run")
        if status == "error":
            judge_excluded += 1
            continue
        if status == "not_run":
            # A retrieval-only run cannot evaluate the judge stage.
            judge_excluded += 1
            continue

        relevant = bool(row.get(judge_relevant_key))
        judge_truth.append(truth)
        judge_pred.append(1 if relevant else 0)
        judge_groups.append(group)

    retrieval_overall = binary_metrics(retrieval_truth, retrieval_pred)
    retrieval_per_group = per_group_metrics(
        retrieval_groups, retrieval_truth, retrieval_pred
    )
    retrieval_macro, retrieval_macro_groups = macro_f1_over_groups(
        retrieval_per_group
    )

    judge_overall = binary_metrics(judge_truth, judge_pred)
    judge_per_group = per_group_metrics(judge_groups, judge_truth, judge_pred)
    judge_macro, judge_macro_groups = macro_f1_over_groups(judge_per_group)
    judge_total = len(judge_truth) + judge_excluded

    return {
        "threshold": round(float(threshold), 6),
        "retrieval_only": {
            "overall": retrieval_overall,
            "per_sdg": retrieval_per_group,
            "macro_f1": retrieval_macro,
            "macro_f1_groups": retrieval_macro_groups,
        },
        "retrieve_judge": {
            "overall": judge_overall,
            "per_sdg": judge_per_group,
            "macro_f1": judge_macro,
            "macro_f1_groups": judge_macro_groups,
            "judge_excluded": judge_excluded,
            "judge_coverage": round(
                _safe_ratio(len(judge_truth), judge_total), 6
            ),
        },
    }


def threshold_grid(
    rows: Sequence[dict[str, Any]],
    thresholds: Iterable[float],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Evaluate every threshold in ``thresholds`` against the same rows."""
    return [
        evaluate_threshold(rows, threshold, **kwargs)
        for threshold in thresholds
    ]


def select_threshold(
    grid: Sequence[dict[str, Any]],
    *,
    method: str = "retrieval_only",
) -> dict[str, Any] | None:
    """Pick the threshold with the best macro-F1, then accuracy, then higher t.

    Macro-F1 is averaged only over SDGs that contain both positive and negative
    rows in the sample. If no SDG does, the overall positive-class F1 is used
    instead so the selection still returns a result.
    """
    if not grid:
        return None

    def score(entry: dict[str, Any]) -> tuple[float, float, float]:
        block = entry[method]
        primary = float(block.get("macro_f1") or 0.0)
        if int(block.get("macro_f1_groups") or 0) == 0:
            primary = float(block["overall"].get("f1") or 0.0)
        accuracy = float(block["overall"].get("accuracy") or 0.0)
        # Higher threshold wins a tie: it is more conservative and sends fewer
        # candidates to the judge.
        return (primary, accuracy, float(entry["threshold"]))

    return max(grid, key=score)


def metrics_at(
    grid: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Return the grid entry whose threshold matches ``threshold``."""
    for entry in grid:
        if abs(float(entry["threshold"]) - float(threshold)) < 1e-9:
            return entry
    return None
