"""Pure index-partition helpers for parallel chunking.

No PyPhi imports: these decide how item indices are grouped into chunks,
either evenly (count-balanced) or by estimated cost (weight-balanced).
"""

from __future__ import annotations

import heapq
import math

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
