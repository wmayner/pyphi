"""Bridge the shared perf harness into ASV benchmarks.

ASV checks the project out into its env's ``project/`` dir and runs benchmarks
from ``benchmarks/benchmarks/``. The perf harness lives at the repo root under
``test/golden/``, which is not an installed package, so we put the repo root on
``sys.path`` before importing it, then re-export its names.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.golden.perf import (  # noqa: E402
    FIXTURES_BY_NAME,
    FRAMES,
    GRAINS,
    applies,
    build_system,
    count_calls,
    run_grain,
)

__all__ = [
    "FIXTURES_BY_NAME",
    "FRAMES",
    "GRAINS",
    "applies",
    "build_system",
    "count_calls",
    "run_grain",
]
