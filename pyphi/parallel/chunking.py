"""Pure index-partition helpers for parallel chunking.

No PyPhi imports: these decide how item indices are grouped into chunks,
either evenly (count-balanced) or by estimated cost (weight-balanced).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any

_EPS = 1e-12


def even_partition(n: int, k: int) -> list[list[int]]:
    """Split ``range(n)`` into ``min(k, n)`` contiguous, near-equal bins."""
    k = max(1, min(k, n))
    base, extra = divmod(n, k)
    bins: list[list[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        bins.append(list(range(start, start + size)))
        start += size
    return bins


def cost_balanced_partition(weights: list[float], k: int) -> list[list[int]]:
    """Greedily LPT-pack item indices into ``min(k, n)`` cost-balanced bins.

    Sorts indices by weight descending and assigns each to the currently
    lightest bin. Non-positive / non-finite weights are clamped to a small
    epsilon so every item still lands in exactly one bin.
    """
    n = len(weights)
    k = max(1, min(k, n))
    bins: list[list[int]] = [[] for _ in range(k)]
    heap = [(0.0, i) for i in range(k)]  # (accumulated weight, bin index)
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    for idx in order:
        w = weights[idx]
        if not math.isfinite(w) or w <= 0.0:
            w = _EPS
        acc, b = heapq.heappop(heap)
        bins[b].append(idx)
        heapq.heappush(heap, (acc + w, b))
    return bins


def iter_chunks(
    materialized: Sequence[Sequence[Any]],
    chunksize: int,
    num_workers: int,
    size_func: Callable[[Any], float] | None = None,
) -> Iterator[tuple]:
    """Yield chunk tuples for parallel dispatch.

    Each yielded tuple holds one list per input sequence, index-aligned
    across sequences. Indices are grouped into ``max(ceil(n / chunksize),
    num_workers)`` bins — evenly, or cost-balanced when ``size_func``
    estimates per-item cost from the first sequence's items. ``n`` is the
    length of the shortest sequence, so ragged inputs truncate as
    ``zip(strict=False)`` would.

    Parameters
    ----------
    materialized : sequence of sequences
        The item sequences to chunk; the first is the primary axis.
    chunksize : int
        Target number of items per chunk.
    num_workers : int
        Lower bound on the number of chunks, so a small workload still
        spreads across available workers.
    size_func : callable, optional
        Estimated cost of one primary-axis item. If None, chunks are
        count-balanced.
    """
    if not materialized or not materialized[0]:
        return
    n = min(len(it) for it in materialized)
    k = max(math.ceil(n / chunksize), num_workers)
    if size_func is not None:
        weights = [size_func(materialized[0][i]) for i in range(n)]
        index_bins = cost_balanced_partition(weights, k)
    else:
        index_bins = even_partition(n, k)
    for indices in index_bins:
        if not indices:
            continue
        yield tuple([it[i] for i in indices] for it in materialized)
