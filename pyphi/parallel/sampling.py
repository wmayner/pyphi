"""Cost-sampling chunksize calculation for the Scheduler Protocol.

Samples up to four items spread across the iterable (positions 0, N/4, N/2,
3N/4 for known-length sequences; first four for unknown-length generators),
times them inline, and computes a target chunksize that aims for roughly
``target_seconds`` of wall time per chunk. The sampled results are returned
with their positions so callers can reuse them instead of computing the
sampled items a second time.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
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
) -> tuple[int, Iterator[Any], list[tuple[int, Any]]]:
    """Return ``(chunksize, items_iterator, sampled)`` for a workload.

    The returned iterator yields all original items including the ones used
    for sampling. ``sampled`` holds one ``(position, result)`` pair per
    sampled item (positions index into the returned iterator's order), so
    callers can reuse the results already computed during sampling. If
    ``explicit_chunksize`` is provided, sampling is skipped entirely and
    ``sampled`` is empty.
    """
    if explicit_chunksize is not None:
        return explicit_chunksize, iter(items), []

    if hasattr(items, "__len__"):
        total: int = len(items)  # type: ignore[arg-type]
        if total < sequential_threshold or fn is None:
            return 1, iter(items), []
        if total < sample_size:
            return 1, iter(items), []
        return _sample_known_length(items, total, fn, sample_size, target_seconds)

    return _sample_unknown_length(items, fn, sample_size, target_seconds)


def _sample_known_length(
    items: Iterable[Any],
    total: int,
    fn: Callable[[Any], Any],
    sample_size: int,
    target_seconds: float,
) -> tuple[int, Iterator[Any], list[tuple[int, Any]]]:
    items_list = list(items)
    positions = [int(i * total / sample_size) for i in range(sample_size)]
    elapsed, results = _time_samples(fn, [items_list[p] for p in positions])
    chunksize = _chunksize_from_timing(elapsed, sample_size, target_seconds)
    return chunksize, iter(items_list), list(zip(positions, results, strict=True))


def _sample_unknown_length(
    items: Iterable[Any],
    fn: Callable[[Any], Any] | None,
    sample_size: int,
    target_seconds: float,
) -> tuple[int, Iterator[Any], list[tuple[int, Any]]]:
    iterator = iter(items)
    sampled_items: list[Any] = []
    for _ in range(sample_size):
        try:
            sampled_items.append(next(iterator))
        except StopIteration:
            break
    if fn is None or not sampled_items:
        return 1, chain(sampled_items, iterator), []
    elapsed, results = _time_samples(fn, sampled_items)
    chunksize = _chunksize_from_timing(elapsed, len(sampled_items), target_seconds)
    return chunksize, chain(sampled_items, iterator), list(enumerate(results))


def _time_samples(
    fn: Callable[[Any], Any], samples: list[Any]
) -> tuple[float, list[Any]]:
    start = time.perf_counter()
    results = [fn(item) for item in samples]
    return time.perf_counter() - start, results


def _chunksize_from_timing(elapsed: float, n: int, target_seconds: float) -> int:
    if elapsed <= 0:
        return 1
    mean_per_item = elapsed / n
    return max(1, int(target_seconds / mean_per_item))


@dataclass
class WorkloadPlan:
    """A cost-sampled workload, ready for chunked dispatch.

    ``items`` and ``more_items`` are the columns still to be computed;
    ``reducer`` folds any reused sampled results into the reduction.
    """

    chunksize: int
    sequential_threshold: int
    items: list[Any]
    more_items: tuple[list[Any], ...]
    reducer: Callable[[Iterable[Any]], Any]


def plan_workload(
    fn: Callable[..., Any],
    items: list[Any],
    more_items: tuple[Iterable[Any], ...],
    *,
    map_kwargs: dict[str, Any],
    chunking: Any,
    ordered: bool,
    shortcircuit_active: bool,
    reducer: Callable[[Iterable[Any]], Any],
) -> WorkloadPlan:
    """Cost-sample a known-length workload and fold the results into a plan.

    Multi-iterable workloads are zipped into argument tuples so the sampler
    sees the same call shape as the real map. When collection order is
    unconstrained (``ordered`` is false and no short-circuit predicate is
    active), the sampled items are removed from the returned columns and
    their already-computed results are appended by the returned reducer, so
    no item is computed twice. When order is constrained, the sampled
    results are discarded and the full workload is dispatched, preserving
    the sequential-evaluation prefix semantics.

    A sampled chunksize estimates the number of items per
    ``chunking.target_seconds`` of work, so a workload that fits within one
    such chunk is not worth dispatching; it is folded into the returned
    ``sequential_threshold``. An explicitly configured chunksize governs
    granularity only and leaves the threshold unchanged.
    """
    threshold = chunking.sequential_threshold
    if more_items:
        rows: list[Any] = list(zip(items, *more_items, strict=False))

        def sample_fn(row: Any) -> Any:
            return fn(*row, **map_kwargs)
    else:
        rows = items
        sample_fn = functools.partial(fn, **map_kwargs) if map_kwargs else fn

    chunksize, row_iter, sampled = compute_chunksize(
        rows,
        target_seconds=chunking.target_seconds,
        fn=sample_fn,
        sequential_threshold=threshold,
        explicit_chunksize=chunking.chunksize,
    )
    rows = list(row_iter)
    if chunking.chunksize is None:
        threshold = max(threshold, chunksize + 1)

    if sampled and not ordered and not shortcircuit_active:
        sampled_positions = {position for position, _ in sampled}
        rows = [row for i, row in enumerate(rows) if i not in sampled_positions]
        presampled = [result for _, result in sampled]
        base_reducer = reducer

        def _reuse_reducer(results: Iterable[Any]) -> Any:
            return base_reducer([*results, *presampled])

        reducer = _reuse_reducer

    if more_items:
        columns = (
            [list(column) for column in zip(*rows, strict=True)]
            if rows
            else [[] for _ in range(1 + len(more_items))]
        )
        planned_items, planned_more = columns[0], tuple(columns[1:])
    else:
        planned_items, planned_more = rows, ()
    return WorkloadPlan(chunksize, threshold, planned_items, planned_more, reducer)
