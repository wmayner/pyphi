"""Cost-sampling chunksize calculation for the Scheduler Protocol.

Samples up to four items spread across the iterable (positions 0, N/4, N/2,
3N/4 for known-length sequences; first four for unknown-length generators),
times them inline, and computes a target chunksize that aims for roughly
``target_seconds`` of wall time per chunk.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from itertools import chain
from typing import Any

DEFAULT_SAMPLE_SIZE = 4
DEFAULT_TARGET_SECONDS = 1.0


def compute_chunksize(
    items: Iterable[Any],
    *,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    fn: Callable[[Any], Any] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sequential_threshold: int = 1,
    explicit_chunksize: int | None = None,
) -> tuple[int, Iterator[Any]]:
    """Return ``(chunksize, items_iterator)`` for a workload.

    The returned iterator yields all original items including the ones
    used for sampling. If ``explicit_chunksize`` is provided, sampling is
    skipped entirely.
    """
    if explicit_chunksize is not None:
        return explicit_chunksize, iter(items)

    if hasattr(items, "__len__"):
        total: int = len(items)  # type: ignore[arg-type]
        if total < sequential_threshold or fn is None:
            return 1, iter(items)
        if total < sample_size:
            return 1, iter(items)
        return _sample_known_length(items, total, fn, sample_size, target_seconds)

    return _sample_unknown_length(items, fn, sample_size, target_seconds)


def _sample_known_length(
    items: Iterable[Any],
    total: int,
    fn: Callable[[Any], Any],
    sample_size: int,
    target_seconds: float,
) -> tuple[int, Iterator[Any]]:
    items_list = list(items)
    positions = [int(i * total / sample_size) for i in range(sample_size)]
    samples = [items_list[p] for p in positions]
    elapsed = _time_samples(fn, samples)
    chunksize = _chunksize_from_timing(elapsed, sample_size, target_seconds)
    return chunksize, iter(items_list)


def _sample_unknown_length(
    items: Iterable[Any],
    fn: Callable[[Any], Any] | None,
    sample_size: int,
    target_seconds: float,
) -> tuple[int, Iterator[Any]]:
    iterator = iter(items)
    sampled: list[Any] = []
    for _ in range(sample_size):
        try:
            sampled.append(next(iterator))
        except StopIteration:
            break
    if fn is None or not sampled:
        return 1, chain(sampled, iterator)
    elapsed = _time_samples(fn, sampled)
    chunksize = _chunksize_from_timing(elapsed, len(sampled), target_seconds)
    return chunksize, chain(sampled, iterator)


def _time_samples(fn: Callable[[Any], Any], samples: list[Any]) -> float:
    start = time.perf_counter()
    for item in samples:
        fn(item)
    return time.perf_counter() - start


def _chunksize_from_timing(elapsed: float, n: int, target_seconds: float) -> int:
    if elapsed <= 0:
        return 1
    mean_per_item = elapsed / n
    return max(1, int(target_seconds / mean_per_item))
