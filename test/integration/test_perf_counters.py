"""Deterministic call-count regression gate (cProfile-based).

Exact call counts are pinned in ``test/data/perf/call_counts.json`` for a
bounded, fast subset (each formalism x layer + k-ary + relations-heavy + AC).
The counts are deterministic, so any change fails the build; a legitimate
algorithm change regenerates the pins via ``scripts/gen_perf_counts.py``,
reviewed in the diff like a phi golden. This catches the redundant-work class
of regression (e.g. config.override per partition) that wall-time budgets miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyphi import actual
from pyphi import config
from pyphi import examples
from pyphi.conf import presets
from test.golden.perf import FIXTURES_BY_NAME
from test.golden.perf import FRAMES
from test.golden.perf import count_calls
from test.golden.perf import run_grain

_PINS = json.loads(
    (Path(__file__).parent.parent / "data" / "perf" / "call_counts.json").read_text()
)
_GOLDEN_KEYS = [k for k in _PINS if k != "actual_causation::account"]


def _helper(i: int) -> int:
    return i * 2


def test_count_calls_counts_a_known_frame():
    def thunk():
        return [_helper(i) for i in range(5)]

    counts = count_calls(thunk, [("test_perf_counters", "_helper")])
    assert counts["test_perf_counters:_helper"] == 5


@pytest.mark.parametrize("key", _GOLDEN_KEYS)
def test_call_counts_pinned(key: str) -> None:
    name, grain = key.split("::")
    fixture = FIXTURES_BY_NAME[name]
    counts = count_calls(lambda: run_grain(fixture, grain), FRAMES)
    assert counts == _PINS[key], (
        f"{key} call counts changed from the pins. If this is a deliberate "
        f"algorithm change, regenerate: uv run python scripts/gen_perf_counts.py"
    )


def test_call_counts_pinned_actual_causation() -> None:
    with config.override(**presets.iit3):
        transition = actual.Transition(
            examples.actual_causation_substrate(), (1, 0), (1, 0), (0, 1), (0, 1)
        )
        counts = count_calls(lambda: actual.account(transition), FRAMES)
        assert counts == _PINS["actual_causation::account"], (
            "AC account call counts changed from the pins. If deliberate, "
            "regenerate: uv run python scripts/gen_perf_counts.py"
        )
