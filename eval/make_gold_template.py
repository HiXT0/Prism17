#!/usr/bin/env python
"""Create a blank human-annotation template from one or more audit JSON files.

The template contains the model's findings and empty ``reviewer_*`` columns.
It contains no gold labels. Fill the reviewer columns by hand, following
``data/gold/ANNOTATION_GUIDELINES.md``, then score with
``eval/score_benchmark.py``.

Run:
    python eval/make_gold_template.py outputs/sjs.phase2.json \
        --out data/gold/annotation_template.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.benchmark import build_annotation_rows, write_annotation_csv  # noqa: E402
from prism17.config import load_config, resolve_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a blank CSV annotation template from Prism17 audit JSON."
        )
    )
    parser.add_argument("audits", nargs="+", help="Audit JSON files")
    parser.add_argument(
        "--out",
        default="data/gold/annotation_template.csv",
        help="Output CSV path (default: data/gold/annotation_template.csv)",
    )
    parser.add_argument(
        "--prefix",
        default="ann",
        help="Annotation id prefix (default: ann)",
    )
    parser.add_argument(
        "--scored-only",
        action="store_true",
        help="Include only scored findings",
    )
    parser.add_argument(
        "--include-not-relevant",
        action="store_true",
        help="Also include candidates the judge marked not relevant",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing template",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config()
        out_path = resolve_path(args.out, config)
        if out_path.exists() and not args.force:
            raise ValueError(
                f"{out_path} already exists; pass --force to overwrite it"
            )

        rows: list[dict[str, str]] = []
        for index, audit in enumerate(args.audits, start=1):
            audit_path = resolve_path(audit, config)
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            prefix = f"{args.prefix}-{audit_path.stem}"
            if len(args.audits) > 1:
                prefix = f"{prefix}-{index}"
            rows.extend(
                build_annotation_rows(
                    payload,
                    audit_path=str(audit_path),
                    prefix=prefix,
                    include_unscored=not args.scored_only,
                    include_not_relevant=args.include_not_relevant,
                )
            )

        write_annotation_csv(rows, out_path)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(rows)} annotation rows to {out_path}")
    print(
        "The reviewer_* columns are blank: this template contains no gold "
        "labels."
    )
    print(
        "Fill them by hand, then run: python eval/score_benchmark.py "
        f"{out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
