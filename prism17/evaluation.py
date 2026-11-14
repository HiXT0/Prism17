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

from .config import display_path


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
        "path": display_path(resolved),
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


# ---------------------------------------------------------------------------
# Agreement and classification diagnostics
# ---------------------------------------------------------------------------


def _ordered_labels(values: Iterable[Any], labels: Sequence[Any] | None) -> list[Any]:
    if labels is not None:
        return list(labels)
    unique = {value for value in values if value is not None}
    try:
        return sorted(unique, key=lambda value: float(value))
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return a labelled confusion matrix (rows: true, columns: predicted)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    ordered = _ordered_labels([*y_true, *y_pred], labels)
    index = {label: position for position, label in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    skipped = 0
    for truth, prediction in zip(y_true, y_pred):
        if truth not in index or prediction not in index:
            skipped += 1
            continue
        matrix[index[truth]][index[prediction]] += 1

    return {
        "labels": ordered,
        "matrix": matrix,
        "skipped": skipped,
    }


def cohen_kappa(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> float | None:
    """Return Cohen's kappa, or ``None`` when there is nothing to compare."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None

    ordered = _ordered_labels([*y_true, *y_pred], labels)
    index = {label: position for position, label in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    for truth, prediction in zip(y_true, y_pred):
        if truth in index and prediction in index:
            matrix[index[truth]][index[prediction]] += 1

    total = sum(sum(row) for row in matrix)
    if total == 0:
        return None
    observed = sum(matrix[i][i] for i in range(len(ordered))) / total
    expected = 0.0
    for i in range(len(ordered)):
        row_total = sum(matrix[i])
        column_total = sum(matrix[row][i] for row in range(len(ordered)))
        expected += (row_total / total) * (column_total / total)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 6)


def quadratic_weighted_kappa(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[int] = (1, 2, 3, 4),
) -> float | None:
    """Return quadratic weighted kappa for ordered rubric levels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None

    positions = {value: index for index, value in enumerate(labels)}
    size = len(labels)
    observed = [[0 for _ in range(size)] for _ in range(size)]
    for truth, prediction in zip(y_true, y_pred):
        if truth not in positions or prediction not in positions:
            continue
        observed[positions[truth]][positions[prediction]] += 1

    total = sum(sum(row) for row in observed)
    if total == 0:
        return None
    row_totals = [sum(row) for row in observed]
    column_totals = [
        sum(observed[row][column] for row in range(size))
        for column in range(size)
    ]
    expected = [
        [
            (row_totals[row] * column_totals[column]) / total
            for column in range(size)
        ]
        for row in range(size)
    ]

    numerator = 0.0
    denominator = 0.0
    for row in range(size):
        for column in range(size):
            weight = ((row - column) ** 2) / ((size - 1) ** 2 or 1)
            numerator += weight * observed[row][column]
            denominator += weight * expected[row][column]
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return round(1.0 - (numerator / denominator), 6)


def mean_absolute_difference(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float | None:
    """Return the mean absolute difference, or ``None`` for no pairs."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None
    total = sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred))
    return round(total / len(y_true), 6)


def categorical_agreement(
    reviewer: Sequence[Any],
    model: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compare two categorical label sequences on their paired entries.

    ``reviewer`` and ``model`` must be aligned. ``None`` marks a missing value
    and is skipped. The result includes the pair count, exact agreement, Cohen's
    kappa, and a confusion matrix.
    """
    if len(reviewer) != len(model):
        raise ValueError("reviewer and model must have the same length")

    pairs = [
        (left, right)
        for left, right in zip(reviewer, model)
        if left is not None and right is not None
    ]
    if not pairs:
        ordered = _ordered_labels([], labels)
        return {
            "n": 0,
            "agreement": None,
            "kappa": None,
            "confusion": {
                "labels": ordered,
                "matrix": [[0 for _ in ordered] for _ in ordered],
                "skipped": 0,
            },
        }

    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    correct = sum(1 for a, b in pairs if a == b)
    return {
        "n": len(pairs),
        "agreement": round(correct / len(pairs), 6),
        "kappa": cohen_kappa(left, right, labels),
        "confusion": confusion_matrix(left, right, labels),
    }


def rubric_agreement(
    reviewer_levels: Sequence[int],
    model_levels: Sequence[int],
) -> dict[str, Any]:
    """Compare reviewer and model rubric levels on paired entries."""
    pairs = [
        (int(reviewer), int(model))
        for reviewer, model in zip(reviewer_levels, model_levels)
        if reviewer is not None and model is not None
    ]
    if not pairs:
        return {
            "n": 0,
            "exact_agreement": None,
            "within_one": None,
            "mean_absolute_difference": None,
            "quadratic_weighted_kappa": None,
            "confusion": confusion_matrix([], [], [1, 2, 3, 4]),
        }

    reviewer_values = [pair[0] for pair in pairs]
    model_values = [pair[1] for pair in pairs]
    exact = sum(1 for a, b in pairs if a == b) / len(pairs)
    within_one = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
    return {
        "n": len(pairs),
        "exact_agreement": round(exact, 6),
        "within_one": round(within_one, 6),
        "mean_absolute_difference": mean_absolute_difference(
            reviewer_values, model_values
        ),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            reviewer_values, model_values
        ),
        "confusion": confusion_matrix(
            reviewer_values, model_values, [1, 2, 3, 4]
        ),
    }
