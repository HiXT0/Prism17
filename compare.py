#!/usr/bin/env python
"""Descriptive cross-company SDG disclosure comparison.

This is a thin command-line wrapper around ``prism17.compare``. The output
describes disclosure counts and rubric levels; it is not a ranking and does not
normalize for report length, sector, or retrieval settings.

Run:
    python compare.py alpha.audit.json beta.audit.json \\
        --labels "Alpha" "Beta" --sdg 13 \\
        --out outputs/compare.json --markdown outputs/compare.md
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.compare import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
