#!/usr/bin/env python
"""Score human annotations against model output.

The script never fills in missing annotations and never invents metrics. Blank
reviewer columns produce ``status: "no_annotations"`` with null metrics.
Invalid reviewer values produce ``status: "invalid_annotations"``. Reviewer
values without a model counterpart produce ``status: "no_pairs"``.

Run:
    python eval/score_benchmark.py data/gold/annotations.csv \\
        --out eval/benchmark.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.benchmark import score_csv, write_report  # noqa: E402
from prism17.evaluation import build_provenance, file_info  # noqa: E402

BENCHMARK_DATASET = {
    "name": "Prism17 human annotation benchmark",
    "version": "0.1",
    "description": (
        "Independent reviewer labels for relevance, rubric level, quote "
        "verification, and semantic support."
    ),
}


def _print_summary(metrics: dict) -> None:
    print(f"Rows: {metrics['rows_total']}")
    print(
        f"Pairs: relevance={metrics['pairs']['relevance']}, "
        f"rubric={metrics['pairs']['rubric']}, "
        f"quote_verification={metrics['pairs']['quote_verification']}, "
        f"semantic_support={metrics['pairs']['semantic_support']}"
    )

    if metrics["status"] == "no_annotations":
        print("No reviewer annotations found; no metrics were computed.")
        return
    if metrics["status"] == "invalid_annotations":
        print(
            "No valid reviewer annotations were found; check the accepted "
            "values in docs/ANNOTATION_GUIDELINES.md. No metrics were "
            "computed."
        )
        return
    if metrics["status"] == "no_pairs":
        print(
            "Reviewer annotations were found, but no model value was "
            "available for a paired metric. No metrics were computed."
        )
        return

    relevance = metrics.get("relevance")
    if relevance:
        print(
            "Relevance: "
            f"accuracy={relevance['accuracy']:.3f} "
            f"precision={relevance['precision']:.3f} "
            f"recall={relevance['recall']:.3f} "
            f"f1={relevance['f1']:.3f}"
        )
    rubric = metrics.get("rubric")
    if rubric:
        print(
            "Rubric: "
            f"exact={rubric['exact_agreement']:.3f} "
            f"within_one={rubric['within_one']:.3f} "
            f"qwk={rubric['quadratic_weighted_kappa']}"
        )
    quote = metrics.get("quote_verification")
    if quote:
        print(
            "Quote verification: "
            f"accuracy={quote['accuracy']:.3f} "
            f"precision={quote['precision']:.3f} "
            f"recall={quote['recall']:.3f} "
            f"f1={quote['f1']:.3f}"
        )
    semantic = metrics.get("semantic_support")
    if semantic:
        print(
            "Semantic support: "
            f"agreement={semantic['agreement']} "
            f"kappa={semantic['kappa']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score human annotations against Prism17 model outputs."
    )
    parser.add_argument("csv_path", help="Gold annotation CSV")
    parser.add_argument(
        "--out",
        default="eval/benchmark.json",
        help="Output benchmark JSON path",
    )
    args = parser.parse_args(argv)

    try:
        metrics = score_csv(args.csv_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    notes = []
    if metrics["status"] == "no_annotations":
        notes.append(
            "No reviewer annotations were present. No benchmark metrics were "
            "computed; do not report numbers until the reviewer_* columns are "
            "filled."
        )
    elif metrics["status"] == "invalid_annotations":
        notes.append(
            "Reviewer values were present but invalid. No benchmark metrics "
            "were computed; correct the values before reporting numbers."
        )
    elif metrics["status"] == "no_pairs":
        notes.append(
            "Reviewer values were present but no model value was available for "
            "a paired metric. No benchmark metrics were computed."
        )
    else:
        notes.append(
            "Metrics are computed only from rows with both a valid reviewer "
            "value and a model value."
        )

    document = {
        "provenance": build_provenance(
            dataset=BENCHMARK_DATASET,
            files={"annotations": file_info(args.csv_path)},
            config={"status": metrics["status"]},
            sample={"rows_total": metrics["rows_total"]},
            notes=notes,
        ),
        "metrics": metrics,
    }

    _print_summary(metrics)
    path = write_report(args.out, document)
    print(f"Wrote benchmark report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
