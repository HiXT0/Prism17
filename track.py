#!/usr/bin/env python
"""Narrow multi-year SDG tracking over two audit JSON files.

This is a thin command-line wrapper around ``prism17.tracking``. It compares
only quantified, scored findings at or above the configured rubric level and
matches targets by exact SDG target ID. It does not infer a trend.

Run:
    python track.py earlier.audit.json later.audit.json \\
        --earlier-label "Alpha FY2022" --later-label "Alpha FY2023" \\
        --out outputs/alpha.tracking.json --markdown outputs/alpha.tracking.md
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.tracking import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
