#!/usr/bin/env python
"""Build a human-annotation template from one or more audit JSON files.

The reviewer columns are deliberately blank. Do not fill them with model
predictions; they are reserved for independent human labels.

Run:
    python eval/make_gold_template.py outputs/a.audit.json outputs/b.audit.json \\
        --out data/gold/annotations.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.audit_io import load_payload  # noqa: E402
from prism17.benchmark import (  # noqa: E402
    BENCHMARK_FIELDS,
    template_rows,
    write_template,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a blank human-annotation template from audit JSON."
    )
    parser.add_argument("audits", nargs="+", help="Audit JSON files")
    parser.add_argument(
        "--out",
        default="data/gold/annotations.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Company labels"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing gold file"
    )
    args = parser.parse_args(argv)

    try:
        if args.labels and len(args.labels) != len(args.audits):
            raise ValueError("--labels must match the number of audit files")
        payloads = [load_payload(path) for path in args.audits]
        rows = template_rows(
            payloads,
            labels=args.labels,
            paths=args.audits,
        )
        path = write_template(args.out, rows, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} annotation rows to {path}")
    print(f"Columns: {', '.join(BENCHMARK_FIELDS)}")
    print("Fill only the reviewer_* columns, annotator, and annotated_at_utc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
