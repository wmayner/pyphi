"""Tests for cost-sampling chunksize calculation."""

from __future__ import annotations

import time

from pyphi import parallel
from pyphi.parallel.sampling import compute_chunksize


def test_compute_chunksize_below_sequential_threshold_returns_one():
    items = [1, 2, 3]
    chunksize, remainder, sampled = compute_chunksize(
        items, target_seconds=1.0, sequential_threshold=10
    )
    assert chunksize == 1
    assert list(remainder) == items
    assert sampled == []


def test_compute_chunksize_with_explicit_chunksize_skips_sampling():
    items = list(range(100))
    chunksize, remainder, sampled = compute_chunksize(items, explicit_chunksize=5)
    assert chunksize == 5
    assert list(remainder) == items
    assert sampled == []


def test_compute_chunksize_samples_and_chunks(monkeypatch):
    """A 1ms-per-item workload over 1s target chunks at ~1000 items per chunk.

    The sampling timer is injected rather than measured from ``time.sleep`` so
    the derived chunksize is deterministic: real sleep over-sleeps on loaded CI
    runners, which made the ``>= 100`` assertion flaky.
    """
    items = list(range(400))
    # _time_samples() calls perf_counter once before and once after the four
    # samples; simulate 4 items * 1ms = 4ms total -> ~1000 items per chunk.
    ticks = iter([0.0, 0.004])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    chunksize, remainder, sampled = compute_chunksize(
        items, target_seconds=1.0, fn=lambda x: x, sample_size=4
    )
    assert chunksize >= 100
    assert sum(1 for _ in remainder) == len(items)
    assert len(sampled) == 4


def test_compute_chunksize_handles_unknown_length_iterable():
    """Generators without __len__ fall back to first-N samples."""

    def gen():
        yield from range(50)

    chunksize, remainder, sampled = compute_chunksize(
        gen(), target_seconds=0.001, fn=lambda x: x, sample_size=4
    )
    assert chunksize >= 1
    seen = list(remainder)
    assert len(seen) == 50
    assert [position for position, _ in sampled] == [0, 1, 2, 3]


def test_compute_chunksize_returns_one_when_no_fn_provided():
    items = list(range(100))
    chunksize, remainder, sampled = compute_chunksize(items)
    assert chunksize == 1
    assert list(remainder) == items
    assert sampled == []


def test_compute_chunksize_keeps_sampled_positions_and_results():
    """Sampling returns each sampled item's position and result so callers
    can reuse the computed values instead of computing them twice."""
    items = [10, 20, 30, 40, 50, 60, 70, 80]
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    _, remainder, sampled = compute_chunksize(items, fn=fn, sample_size=4)
    assert list(remainder) == items
    assert [items[position] for position, _ in sampled] == calls
    assert all(result == items[position] * 2 for position, result in sampled)


# ============================================================================
# Sampled results are reused, not recomputed
# ============================================================================


def test_sampling_reuses_results_when_dispatch_is_sequential():
    """Sampled items must not be computed a second time when the workload
    then runs sequentially in the parent process."""
    calls = []

    def record(x):
        calls.append(x)
        time.sleep(0.0005)
        return x

    out = parallel.map_reduce(
        record, list(range(8)), parallel=True, backend="local", progress=False
    )
    assert sorted(out) == list(range(8))
    assert len(calls) == 8, "each item must be computed exactly once"


def _record_to_file(x, path):
    with open(path, "a") as f:
        f.write(f"{x}\n")
    time.sleep(0.01)
    return x


def test_sampling_reuses_results_on_parallel_dispatch(tmp_path):
    """Sampled items must not be re-dispatched to workers: each item is
    computed exactly once across the parent and all worker processes."""
    from pyphi.parallel.backends.local_process import LocalProcessScheduler
    from pyphi.parallel.scheduler import ChunkingPolicy

    path = tmp_path / "calls.txt"
    out = LocalProcessScheduler().map_reduce(
        _record_to_file,
        list(range(12)),
        reducer=sorted,
        chunking=ChunkingPolicy(target_seconds=0.02),
        map_kwargs={"path": str(path)},
    )
    assert out == list(range(12))
    lines = path.read_text().splitlines()
    assert sorted(int(line) for line in lines) == list(range(12))


def test_multi_iterable_default_chunksize_samples_and_avoids_per_item_dispatch():
    """A second iterable must not silently disable cost sampling: cheap
    zipped items are sampled (on the real call shape), the sampled chunksize
    keeps the workload sequential in the parent, and no item is computed
    twice."""
    calls = []

    def add(a, b):
        calls.append((a, b))
        time.sleep(0.0005)
        return a + b

    out = parallel.map_reduce(
        add,
        list(range(8)),
        list(range(8)),
        parallel=True,
        backend="local",
        progress=False,
    )
    assert sorted(out) == [2 * i for i in range(8)]
    assert len(calls) == 8, "cheap zipped items must stay in the parent process"
