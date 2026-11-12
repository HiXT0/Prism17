#!/usr/bin/env python
"""Score a filled Prism17 gold annotation CSV.

The report uses deterministic metrics only. It writes null metrics when no
reviewer labels are present; it never fills in placeholder numbers. Agreement
describes disclosure labels, not real-world sustainability performance.

Run:
    python eval/score_benchmark.py data/gold/annotations.csv \
        --out eval/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.benchmark import read_annotation_csv, score_annotations  # noqa: E402
from prism17.config import load_config, resolve_path  # noqa: E402
from prism17.evaluation import (  # noqa: E402
    append_jsonl,
    build_provenance,
    file_info,
)

BENCHMARK_DATASET = {
    "name": "Prism17 gold annotation benchmark",
    "version": "unreleased",
    "label_rule": (
        "reviewer_* columns are human annotations; model_* columns are the "
        "audit output being scored"
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a filled Prism17 gold annotation CSV."
    )
    parser.add_argument("gold", help="Filled annotation CSV")
    parser.add_argument(
        "--out",
        default="eval/benchmark.json",
        help="Output JSON report (default: eval/benchmark.json)",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optionally append one run record to this JSONL log",
    )
    parser.add_argument(
        "--notes",
        action="append",
        default=[],
        help="Add a note to the provenance record (repeatable)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        gold_path = resolve_path(args.gold, config)
        out_path = resolve_path(args.out, config)
        rows = read_annotation_csv(gold_path)
        metrics = score_annotations(rows)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    notes = list(args.notes)
    notes.append(
        "Agreement measures describe disclosure labels only; they do not "
        "verify real-world performance."
    )
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

    document = {
        "provenance": build_provenance(
            dataset=BENCHMARK_DATASET,
            files={"gold": file_info(gold_path)},
            config=config,
            sample={
                "rows_total": metrics["rows_total"],
                "pairs": metrics["pairs"],
            },
            notes=notes,
        ),
        "metrics": metrics,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.jsonl:
        append_jsonl(
            resolve_path(args.jsonl, config),
            {
                "created_at_utc": document["provenance"]["created_at_utc"],
                "gold": str(gold_path),
                "status": metrics["status"],
                "pairs": metrics["pairs"],
            },
        )

    print(f"Rows: {metrics['rows_total']}")
    print(
        "Pairs: "
        f"relevance={metrics['pairs']['relevance']}, "
        f"rubric={metrics['pairs']['rubric']}, "
        f"quote_verification={metrics['pairs']['quote_verification']}, "
        f"semantic_support={metrics['pairs']['semantic_support']}"
    )
    if metrics["status"] == "no_annotations":
        print("No reviewer annotations found; no metrics were computed.")
    elif metrics["status"] == "invalid_annotations":
        print(
            "No valid reviewer annotations were found; check the accepted "
            "values in data/gold/ANNOTATION_GUIDELINES.md. No metrics were "
            "computed."
        )
    elif metrics["status"] == "no_pairs":
        print(
            "Reviewer annotations were found, but no model value was "
            "available for a paired metric. No metrics were computed."
        )
    else:
        if metrics["relevance"]:
            print(
                "Relevance: "
                f"macro-F1={metrics['relevance']['macro_f1']:.3f}, "
                f"accuracy={metrics['relevance']['accuracy']:.3f}"
            )
        if metrics["rubric"]:
            print(
                "Rubric: "
                f"exact={metrics['rubric']['exact_agreement']:.3f}, "
                f"within-one={metrics['rubric']['within_one_agreement']:.3f}, "
                "quadratic-weighted-kappa="
                f"{metrics['rubric']['quadratic_weighted_kappa']:.3f}"
            )
        if metrics["quote_verification"]:
            print(
                "Quote verification: "
                f"macro-F1={metrics['quote_verification']['macro_f1']:.3f}"
            )
        if metrics["semantic_support"]:
            print(
                "Semantic support: "
                "agreement="
                f"{metrics['semantic_support']['observed_agreement']:.3f}, "
                "kappa="
                f"{metrics['semantic_support']['cohen_kappa']:.3f}"
            )
    print(f"Wrote benchmark report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
