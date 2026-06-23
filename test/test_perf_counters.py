"""Deterministic call-count regression gate (cProfile-based)."""

from __future__ import annotations


def _helper(i: int) -> int:
    return i * 2


def test_count_calls_counts_a_known_frame():
    from test.golden.perf import count_calls

    def thunk():
        return [_helper(i) for i in range(5)]

    counts = count_calls(thunk, [("test_perf_counters", "_helper")])
    assert counts["test_perf_counters:_helper"] == 5
