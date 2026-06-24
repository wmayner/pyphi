"""Tests for the pure parallel index-partition helpers."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.parallel.chunking import even_partition


def _is_partition(bins, n):
    flat = [i for b in bins for i in b]
    return sorted(flat) == list(range(n))


def test_even_partition_splits_into_k_bins():
    bins = even_partition(10, 3)
    assert len(bins) == 3
    assert _is_partition(bins, 10)
    assert sorted(len(b) for b in bins) == [3, 3, 4]


def test_even_partition_caps_bins_at_n():
    bins = even_partition(2, 5)
    assert _is_partition(bins, 2)
    assert all(b for b in bins)  # no empty bins


def test_cost_balanced_is_a_partition():
    weights = [5.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    bins = cost_balanced_partition(weights, 2)
    assert _is_partition(bins, len(weights))


def test_cost_balanced_separates_heavy_items():
    # two heavy items must not land in the same bin when k=2
    weights = [10.0, 10.0, 1.0, 1.0]
    bins = cost_balanced_partition(weights, 2)
    placed = {bi for bi, b in enumerate(bins) for i in b if i in (0, 1)}
    assert len(placed) == 2


def test_cost_balanced_clamps_nonpositive_and_nonfinite():
    weights = [0.0, -3.0, float("inf"), float("nan"), 2.0]
    bins = cost_balanced_partition(weights, 2)
    assert _is_partition(bins, len(weights))


def test_k_of_one_returns_single_bin():
    bins = cost_balanced_partition([1.0, 2.0, 3.0], 1)
    assert len(bins) == 1
    assert sorted(bins[0]) == [0, 1, 2]  # all items, one bin (order irrelevant)


@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=200),
    k=st.integers(min_value=1, max_value=64),
)
def test_cost_balanced_partition_property(weights, k):
    bins = cost_balanced_partition(weights, k)
    flat = [i for b in bins for i in b]
    assert sorted(flat) == list(range(len(weights)))  # exact partition
    assert len([b for b in bins if b]) <= min(k, len(weights))
