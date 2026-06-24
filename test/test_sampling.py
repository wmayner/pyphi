"""Tests for cost-sampling chunksize calculation."""

from __future__ import annotations

import time

from pyphi.parallel.sampling import compute_chunksize


def test_compute_chunksize_below_sequential_threshold_returns_one():
    items = [1, 2, 3]
    chunksize, remainder = compute_chunksize(
        items, target_seconds=1.0, sequential_threshold=10
    )
    assert chunksize == 1
    assert list(remainder) == items


def test_compute_chunksize_with_explicit_chunksize_skips_sampling():
    items = list(range(100))
    chunksize, remainder = compute_chunksize(items, explicit_chunksize=5)
    assert chunksize == 5
    assert list(remainder) == items


def test_compute_chunksize_samples_and_chunks():
    """A 1ms-per-item workload over 1s target chunks at ~1000 items per chunk."""
    items = list(range(400))

    def fast_op(x):
        time.sleep(0.001)
        return x

    chunksize, remainder = compute_chunksize(
        items, target_seconds=1.0, fn=fast_op, sample_size=4
    )
    assert chunksize >= 100
    assert sum(1 for _ in remainder) == len(items)


def test_compute_chunksize_handles_unknown_length_iterable():
    """Generators without __len__ fall back to first-N samples."""

    def gen():
        yield from range(50)

    chunksize, remainder = compute_chunksize(
        gen(), target_seconds=0.001, fn=lambda x: x, sample_size=4
    )
    assert chunksize >= 1
    seen = list(remainder)
    assert len(seen) == 50


def test_compute_chunksize_returns_one_when_no_fn_provided():
    items = list(range(100))
    chunksize, remainder = compute_chunksize(items)
    assert chunksize == 1
    assert list(remainder) == items
