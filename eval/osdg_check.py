#!/usr/bin/env python
"""Check SDG relevance on a random sample of OSDG Community Dataset rows.

The check has two stages:

1. Retrieve the top-k UN targets for each sampled text and keep the
   highest-scoring target that belongs to the row's suggested SDG.
2. Optionally ask the local Ollama relevance judge whether that text addresses
   that target.

It then sweeps a retrieval-score threshold over the same rows. Retrieval-only
predictions require the best same-goal score to clear the threshold.
Retrieve-then-judge predictions additionally require a positive judge result.
Per-SDG precision and recall are reported at the suggested threshold.

This is an evaluation of the retrieval and relevance stage only. It is not a
rubric evaluation, and it does not use any DEV or test report. Threshold
suggestions come from OSDG data only.

Run:
    python eval/osdg_check.py --sample-size 200
    python eval/osdg_check.py --sample-size 20 --skip-judge  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.config import get, load_config
from prism17.evaluation import (
    append_jsonl,
    build_provenance,
    default_thresholds,
    file_info,
    select_threshold,
    threshold_grid,
)
from prism17.judge import (
    RELEVANCE_PROMPT_VERSION,
    JudgeUnavailableError,
    RelevanceJudge,
)
from prism17.retrieve import rank_targets_for_texts

LOGGER = logging.getLogger("osdg_check")

OSDG_DATASET = {
    "name": "OSDG Community Dataset (OSDG-CD)",
    "version": "2024.04",
    "zenodo_doi": "10.5281/zenodo.11441197",
    "citation_doi": "10.5281/zenodo.5550238",
    "license": "CC-BY-4.0",
    "label_rule": "positive if labels_positive > labels_negative; ties dropped",
}
DEFAULT_OSDG_PATH = "data/osdg/osdg-community-data-v2024-04-01.csv"
DEFAULT_OUT_DIR = "eval"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 17

CSV_FIELDS = [
    "text_id",
    "doi",
    "sdg",
    "label",
    "labels_positive",
    "labels_negative",
    "agreement",
    "best_target_id",
    "best_score",
    "same_goal_targets",
    "same_goal_scores",
    "judge_status",
    "judge_relevant",
    "judge_confidence",
    "judge_reason",
    "text",
]


# ---------------------------------------------------------------------------
# Data loading and sampling
# ---------------------------------------------------------------------------


def load_osdg(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load OSDG data, drop ties, and attach the binary relevance label."""
    if not path.exists():
        raise FileNotFoundError(
            f"OSDG dataset not found: {path}. Download version 2024.04 from "
            "https://zenodo.org/records/11441197 and place the tab-delimited "
            "file under data/osdg/."
        )

    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "doi": str,
            "text_id": str,
            "text": str,
            "sdg": "Int64",
            "labels_negative": "Int64",
            "labels_positive": "Int64",
            "agreement": float,
        },
    )
    required = {
        "doi",
        "text_id",
        "text",
        "sdg",
        "labels_negative",
        "labels_positive",
        "agreement",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"OSDG dataset is missing columns: {', '.join(sorted(missing))}"
        )

    counts = {
        "rows_total": int(len(frame)),
        "ties_dropped": int(
            (frame["labels_positive"] == frame["labels_negative"]).sum()
        ),
    }
    frame = frame[frame["labels_positive"] != frame["labels_negative"]].copy()
    frame["label"] = (frame["labels_positive"] > frame["labels_negative"]).astype(
        int
    )
    frame = frame.dropna(subset=["text", "sdg"])
    frame = frame[frame["text"].astype(str).str.strip().astype(bool)].copy()
    frame["sdg"] = frame["sdg"].astype(int)
    counts["rows_usable"] = int(len(frame))
    counts["rows_positive"] = int(frame["label"].sum())
    counts["rows_negative"] = int(len(frame) - frame["label"].sum())
    return frame.reset_index(drop=True), counts


def sample_rows(
    frame: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Take a reproducible random sample without replacement."""
    if size <= 0:
        raise ValueError("sample size must be positive")
    if frame.empty:
        raise ValueError("no usable OSDG rows to sample")
    take = min(int(size), len(frame))
    sampled = frame.sample(n=take, random_state=seed)
    return sampled.sort_values(["sdg", "text_id"]).reset_index(drop=True)


def sample_support(sampled: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return positive/negative counts for each suggested SDG in the sample."""
    support: dict[str, dict[str, int]] = {}
    for sdg, group in sampled.groupby("sdg", sort=True):
        support[str(int(sdg))] = {
            "n": int(len(group)),
            "positive": int(group["label"].sum()),
            "negative": int(len(group) - group["label"].sum()),
        }
    return support


def _rows_from_frame(sampled: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sampled.to_dict(orient="records"):
        rows.append(
            {
                "text_id": str(record["text_id"]),
                "doi": str(record["doi"]),
                "text": str(record["text"]).strip(),
                "sdg": int(record["sdg"]),
                "label": int(record["label"]),
                "labels_positive": int(record["labels_positive"]),
                "labels_negative": int(record["labels_negative"]),
                "agreement": float(record["agreement"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Retrieval and judging
# ---------------------------------------------------------------------------


def attach_retrieval(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    top_k: int,
) -> None:
    """Attach the top same-goal target and its score to each row in place."""
    texts = [row["text"] for row in rows]
    ranked = rank_targets_for_texts(
        texts,
        config,
        top_k=top_k,
        cache_prefix="osdg",
    )
    for row, candidates in zip(rows, ranked):
        same_goal = [
            candidate
            for candidate in candidates
            if int(candidate["target_goal"]) == int(row["sdg"])
        ]
        row["same_goal"] = same_goal
        if same_goal:
            best = same_goal[0]
            row["best_target_id"] = str(best["target_id"])
            row["best_target_text"] = str(best["target_text"])
            row["best_score"] = float(best["score"])
            row["judge_status"] = "not_run"
        else:
            row["best_target_id"] = None
            row["best_target_text"] = None
            row["best_score"] = 0.0
            row["judge_status"] = "no_same_goal_candidate"
        row["judge_relevant"] = None
        row["judge_confidence"] = None
        row["judge_reason"] = None
        row["judge_details"] = []


def judge_rows(
    rows: Sequence[dict[str, Any]],
    judge: RelevanceJudge,
    *,
    top_n: int = 1,
) -> None:
    """Judge the top same-goal candidate(s) for each row in place."""
    limit = max(1, int(top_n))
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        same_goal = row.get("same_goal") or []
        if not same_goal:
            row["judge_status"] = "no_same_goal_candidate"
            continue

        details: list[dict[str, Any]] = []
        judgements = []
        for candidate in same_goal[:limit]:
            judgement = judge.judge(
                row["text"],
                target_id=str(candidate["target_id"]),
                target_goal=int(row["sdg"]),
                target_text=str(candidate["target_text"]),
            )
            judgements.append(judgement)
            details.append(
                {
                    "target_id": str(candidate["target_id"]),
                    "score": float(candidate["score"]),
                    "status": judgement.status,
                    "relevant": judgement.relevant,
                    "confidence": judgement.confidence,
                    "reason": judgement.reason,
                    "error": judgement.error,
                }
            )

        row["judge_details"] = details
        successful = [item for item in judgements if item.ok]
        if successful:
            row["judge_status"] = "ok"
            row["judge_relevant"] = any(
                bool(item.relevant) for item in successful
            )
            row["judge_confidence"] = max(
                float(item.confidence) for item in successful
            )
            positive = next(
                (item for item in successful if item.relevant), successful[0]
            )
            row["judge_reason"] = positive.reason
        else:
            row["judge_status"] = "error"
            row["judge_relevant"] = None
            row["judge_confidence"] = None
            row["judge_reason"] = "; ".join(
                item.error or "unknown judge error" for item in judgements
            )

        if index == 1 or index == total or index % 5 == 0:
            print(
                f"  judged {index}/{total} "
                f"(SDG {row['sdg']}, {row['judge_status']})",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Metrics and outputs
# ---------------------------------------------------------------------------


def build_metrics(
    rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    *,
    sample_meta: dict[str, Any],
    retrieval_meta: dict[str, Any],
    judge_meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a metrics document and the flattened threshold grid."""
    grid = threshold_grid(rows, thresholds)
    selected = select_threshold(grid, method="retrieval_only")
    if selected is None:
        raise ValueError("threshold grid is empty")

    best_judge = select_threshold(grid, method="retrieve_judge")
    selected_threshold = float(selected["threshold"])

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("judge_status") or "not_run")
        status_counts[status] = status_counts.get(status, 0) + 1

    metrics = {
        "dataset": OSDG_DATASET,
        "sample": sample_meta,
        "retrieval": retrieval_meta,
        "judge": judge_meta,
        "judge_status_counts": status_counts,
        "selection_rule": (
            "highest macro-F1 over SDGs with both classes, then accuracy, "
            "then higher threshold"
        ),
        "retrieval_only": {
            "selected_threshold": selected_threshold,
            **selected["retrieval_only"],
        },
        "retrieve_judge": {
            "selected_threshold": selected_threshold,
            **selected["retrieve_judge"],
            "best_threshold": (
                float(best_judge["threshold"]) if best_judge else None
            ),
            "best": best_judge["retrieve_judge"] if best_judge else None,
        },
    }
    return metrics, grid


def _flatten_grid_entry(entry: dict[str, Any]) -> dict[str, Any]:
    retrieval = entry["retrieval_only"]
    judge = entry["retrieve_judge"]
    return {
        "threshold": entry["threshold"],
        "retrieval_n": retrieval["overall"]["n"],
        "retrieval_accuracy": retrieval["overall"]["accuracy"],
        "retrieval_precision": retrieval["overall"]["precision"],
        "retrieval_recall": retrieval["overall"]["recall"],
        "retrieval_f1": retrieval["overall"]["f1"],
        "retrieval_macro_f1": retrieval["macro_f1"],
        "retrieval_macro_f1_groups": retrieval["macro_f1_groups"],
        "judge_n": judge["overall"]["n"],
        "judge_accuracy": judge["overall"]["accuracy"],
        "judge_precision": judge["overall"]["precision"],
        "judge_recall": judge["overall"]["recall"],
        "judge_f1": judge["overall"]["f1"],
        "judge_macro_f1": judge["macro_f1"],
        "judge_macro_f1_groups": judge["macro_f1_groups"],
        "judge_excluded": judge["judge_excluded"],
        "judge_coverage": judge["judge_coverage"],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _result_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        same_goal = row.get("same_goal") or []
        output.append(
            {
                "text_id": row["text_id"],
                "doi": row["doi"],
                "sdg": row["sdg"],
                "label": row["label"],
                "labels_positive": row["labels_positive"],
                "labels_negative": row["labels_negative"],
                "agreement": row["agreement"],
                "best_target_id": row.get("best_target_id"),
                "best_score": row.get("best_score"),
                "same_goal_targets": ";".join(
                    str(candidate["target_id"]) for candidate in same_goal
                ),
                "same_goal_scores": ";".join(
                    f"{float(candidate['score']):.6f}" for candidate in same_goal
                ),
                "judge_status": row.get("judge_status"),
                "judge_relevant": row.get("judge_relevant"),
                "judge_confidence": row.get("judge_confidence"),
                "judge_reason": row.get("judge_reason"),
                "text": row["text"],
            }
        )
    return output


def write_outputs(
    *,
    out_dir: Path,
    rows: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    grid: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": out_dir / "osdg_check_results.csv",
        "metrics_json": out_dir / "osdg_check_metrics.json",
        "grid_csv": out_dir / "osdg_check_grid.csv",
        "provenance_json": out_dir / "osdg_check_provenance.json",
    }

    _write_csv(paths["results_csv"], _result_rows(rows), CSV_FIELDS)
    paths["metrics_json"].write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        paths["grid_csv"],
        [_flatten_grid_entry(entry) for entry in grid],
        list(_flatten_grid_entry(grid[0]).keys()),
    )
    paths["provenance_json"].write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _config_excerpt(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": config.get("_config_path"),
        "embedding_model": get(
            config, "retrieval", "embedding_model", default=None
        ),
        "query_prefix": get(config, "retrieval", "query_prefix", default=""),
        "configured_top_k": get(config, "retrieval", "top_k", default=None),
        "configured_score_threshold": get(
            config, "retrieval", "score_threshold", default=None
        ),
        "llm_model": get(config, "llm", "model", default=None),
        "llm_temperature": get(config, "llm", "temperature", default=None),
        "relevance_prompt_version": RELEVANCE_PROMPT_VERSION,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check SDG relevance retrieval and judging on a random OSDG "
            "Community Dataset sample."
        )
    )
    parser.add_argument(
        "--osdg",
        default=DEFAULT_OSDG_PATH,
        help="Path to the tab-delimited OSDG 2024.04 file",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help="Directory for result files"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to sample (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random sample seed"
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Targets retrieved per text"
    )
    parser.add_argument(
        "--model", default=None, help="Ollama model override for the judge"
    )
    parser.add_argument(
        "--judge-top-n",
        type=int,
        default=1,
        help="Same-goal candidates to judge per row (default: 1)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run retrieval only; retrieve+judge metrics will be empty",
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.0, help="Grid start"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.8, help="Grid end"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01, help="Grid step"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    osdg_path = Path(args.osdg)
    if not osdg_path.is_absolute():
        osdg_path = PROJECT_ROOT / osdg_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    try:
        config = load_config(args.config)
        frame, counts = load_osdg(osdg_path)
        sampled = sample_rows(frame, args.sample_size, args.seed)
        thresholds = default_thresholds(
            args.threshold_min, args.threshold_max, args.threshold_step
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    rows = _rows_from_frame(sampled)
    retrieval_cfg = get(config, "retrieval", default={}) or {}
    top_k = int(args.top_k if args.top_k is not None else retrieval_cfg.get("top_k", 3))
    model = str(args.model or get(config, "llm", "model", default="qwen2.5:7b-instruct"))

    LOGGER.info(
        "Loaded OSDG %s: %s rows, %s ties dropped, %s usable",
        OSDG_DATASET["version"],
        counts["rows_total"],
        counts["ties_dropped"],
        counts["rows_usable"],
    )
    LOGGER.info(
        "Sampled %s rows (seed=%s, positive=%s, negative=%s)",
        len(rows),
        args.seed,
        sum(row["label"] for row in rows),
        sum(1 - row["label"] for row in rows),
    )

    attach_retrieval(rows, config, top_k=top_k)
    LOGGER.info(
        "Retrieved top-%s targets; %s rows have a same-goal candidate",
        top_k,
        sum(1 for row in rows if row.get("same_goal")),
    )

    judge_meta = {
        "enabled": not args.skip_judge,
        "model": model,
        "prompt_version": RELEVANCE_PROMPT_VERSION,
        "top_n": max(1, int(args.judge_top_n)),
    }
    if not args.skip_judge:
        judge = RelevanceJudge(config, model=model)
        try:
            judge.ensure_available()
            judge_rows(rows, judge, top_n=args.judge_top_n)
        except JudgeUnavailableError as exc:
            LOGGER.error("%s", exc)
            return 2
    else:
        LOGGER.warning(
            "Skipping the relevance judge; retrieve+judge metrics will be empty"
        )

    sample_meta = {
        "size_requested": int(args.sample_size),
        "size_used": len(rows),
        "seed": int(args.seed),
        "positive": sum(row["label"] for row in rows),
        "negative": sum(1 - row["label"] for row in rows),
        "per_sdg": sample_support(sampled),
    }
    retrieval_meta = {
        "top_k": top_k,
        "focus_goals": None,
        "same_goal_candidate_rows": sum(
            1 for row in rows if row.get("same_goal")
        ),
    }

    metrics, grid = build_metrics(
        rows,
        thresholds,
        sample_meta=sample_meta,
        retrieval_meta=retrieval_meta,
        judge_meta=judge_meta,
    )

    files = {"labels": file_info(osdg_path)}
    provenance = build_provenance(
        dataset=OSDG_DATASET,
        files=files,
        config=_config_excerpt(config),
        sample={
            "size_requested": int(args.sample_size),
            "size_used": len(rows),
            "seed": int(args.seed),
            "counts": counts,
            "per_sdg": sample_support(sampled),
        },
        notes=[
            "Threshold suggestions come from OSDG data only. No DEV or test "
            "report was used to tune prompts or thresholds.",
            "Per-SDG metrics use sampled rows whose suggested SDG is that goal; "
            "OSDG validates one suggested label per text.",
            "Retrieve+judge uses the top same-goal candidate(s) per row and "
            "excludes rows whose surfaced candidate has no successful judge "
            "result.",
        ],
    )

    paths = write_outputs(
        out_dir=out_dir,
        rows=rows,
        metrics=metrics,
        grid=grid,
        provenance=provenance,
    )
    append_jsonl(
        out_dir / "evaluation_runs.log",
        {
            "created_at_utc": provenance["created_at_utc"],
            "evaluation": "osdg_check",
            "dataset": OSDG_DATASET,
            "files": files,
            "sample": provenance["sample"],
            "outputs": {key: str(value) for key, value in paths.items()},
        },
    )

    retrieval_only = metrics["retrieval_only"]
    retrieve_judge = metrics["retrieve_judge"]
    print(
        f"\nOSDG {OSDG_DATASET['version']} | sample {len(rows)} "
        f"(seed {args.seed}) | top_k {top_k}"
    )
    print(
        f"Suggested retrieval threshold: {retrieval_only['selected_threshold']:.2f}"
    )
    print(
        "  retrieval-only: "
        f"accuracy={retrieval_only['overall']['accuracy']:.3f} "
        f"precision={retrieval_only['overall']['precision']:.3f} "
        f"recall={retrieval_only['overall']['recall']:.3f} "
        f"macro-F1={retrieval_only['macro_f1']:.3f} "
        f"({retrieval_only['macro_f1_groups']} SDGs)"
    )
    if judge_meta["enabled"]:
        print(
            "  retrieve+judge: "
            f"accuracy={retrieve_judge['overall']['accuracy']:.3f} "
            f"precision={retrieve_judge['overall']['precision']:.3f} "
            f"recall={retrieve_judge['overall']['recall']:.3f} "
            f"macro-F1={retrieve_judge['macro_f1']:.3f} "
            f"(coverage={retrieve_judge['judge_coverage']:.2f})"
        )
        print(
            f"  best retrieve+judge threshold: "
            f"{retrieve_judge['best_threshold']:.2f}"
        )
    print("Per-SDG precision/recall at the suggested threshold:")
    for sdg, values in retrieval_only["per_sdg"].items():
        print(
            f"  SDG {sdg:>2}: n={values['n']:>3} "
            f"precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f}"
        )
    print("\nWrote:")
    for path in paths.values():
        print(f"  {path}")
    print(f"  {out_dir / 'evaluation_runs.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
