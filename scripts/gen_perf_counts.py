"""Regenerate test/data/perf/call_counts.json (the perf-counter gate pins).

Run after a deliberate algorithm change that alters call structure:

    uv run python scripts/gen_perf_counts.py

Review the JSON diff exactly like a phi golden.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyphi import actual
from pyphi import config
from pyphi import examples
from pyphi.conf import presets
from test.golden.perf import FIXTURES_BY_NAME
from test.golden.perf import FRAMES
from test.golden.perf import count_calls
from test.golden.perf import run_grain

# (fixture_name, grain) — the bounded, fast gate subset: every formalism x layer
# plus a k-ary fixture and a relations-heavy 4.0 fixture.
GATE_SUBSET = [
    ("basic_iit3_emd", "sia"),
    ("basic_iit4_2023", "sia"),
    ("basic_iit4_2026", "sia"),
    ("basic_iit4_2023", "mechanism_mips"),
    ("basic_iit4_2023", "repertoires"),
    ("basic_iit4_2023", "phi_structure"),
    ("multivalued_k3_tiny_iit4_2023", "sia"),
    ("rule110_iit4_2023", "phi_structure"),
]

_OUT = (
    Path(__file__).resolve().parents[1] / "test" / "data" / "perf" / "call_counts.json"
)


def _account_counts() -> dict[str, int]:
    """Actual Causation gate entry (outside the golden zoo)."""

    def _run_ac() -> None:
        transition = actual.Transition(
            examples.actual_causation_substrate(), (1, 0), (1, 0), (0, 1), (0, 1)
        )
        actual.account(transition)

    counts: dict[str, int] = {}
    with config.override(**presets.iit3):
        counts = count_calls(_run_ac, FRAMES)
    return counts


def main() -> None:
    pins: dict[str, dict[str, int]] = {}
    for name, grain in GATE_SUBSET:
        fixture = FIXTURES_BY_NAME[name]
        counts = count_calls(lambda: run_grain(fixture, grain), FRAMES)  # noqa: B023
        pins[f"{name}::{grain}"] = counts
        print(f"{name}::{grain}: {counts}")  # noqa: T201
    pins["actual_causation::account"] = _account_counts()
    print(f"actual_causation::account: {pins['actual_causation::account']}")  # noqa: T201
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
