"""Environment generators for matching.

A generator is a *world distribution*: a mapping from sensory-interface states
(length-``n`` 0/1 tuples) to probabilities summing to 1, suitable as the
``world_distribution`` of :class:`pyphi.matching.MatchingAnalysis`. Distributions
are computed exactly over the sensory interface. Keys align positionally with the
caller's ``sensory_indices``; "contiguous" refers to that ordering.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

Distribution = dict[tuple[int, ...], float]


def _normalize(dist: Distribution) -> Distribution:
    """Validate non-negativity, drop zero-mass states, renormalize to sum 1."""
    if any(p < 0 for p in dist.values()):
        raise ValueError("probabilities must be non-negative")
    total = float(sum(dist.values()))
    if total <= 0:
        raise ValueError("distribution has zero total mass")
    return {state: p / total for state, p in dist.items() if p > 0}


def _check_p(p: float) -> None:
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1]; got {p}")


def segment(n: int, length: int, p: float) -> Distribution:
    """A run of ``length`` contiguous units at a uniformly random location.

    With probability ``p`` a segment is present (its location uniform over the
    ``n - length + 1`` start positions); with probability ``1 - p`` no unit is
    active (all-off).
    """
    _check_p(p)
    if not 1 <= length <= n:
        raise ValueError(f"length must be in [1, n]={n}; got {length}")
    positions = n - length + 1
    dist: Distribution = defaultdict(float)
    dist[tuple([0] * n)] += 1 - p
    for start in range(positions):
        state = [0] * n
        for i in range(start, start + length):
            state[i] = 1
        dist[tuple(state)] += p / positions
    return _normalize(dict(dist))


def point(n: int, p: float) -> Distribution:
    """A single unit active at a uniformly random location with probability ``p``."""
    return segment(n, 1, p)


def noise(n: int, p: float) -> Distribution:
    """Each unit independently active with probability ``p`` (product Bernoulli).

    ``p = 0.5`` yields the uniform "structureless world".
    """
    _check_p(p)
    dist: Distribution = {}
    for state in itertools.product((0, 1), repeat=n):
        prob = 1.0
        for s in state:
            prob *= p if s else (1 - p)
        dist[state] = prob
    return _normalize(dist)


def _shared_n(distributions) -> int:
    sizes = {len(next(iter(d))) for d in distributions}
    if len(sizes) != 1:
        raise ValueError(f"all distributions must share the same n; got {sizes}")
    return sizes.pop()


def superpose(*distributions: Distribution) -> Distribution:
    """Independent activation of each generator, combined by elementwise OR.

    Each input distribution is drawn independently; a unit is active in the
    result iff any generator activates it. Computed exactly over the product of
    the inputs' supports.
    """
    if not distributions:
        raise ValueError("superpose requires at least one distribution")
    n = _shared_n(distributions)
    result: Distribution = defaultdict(float)
    for combo in itertools.product(*(d.items() for d in distributions)):
        prob = 1.0
        merged = [0] * n
        for state, state_prob in combo:
            prob *= state_prob
            for i, s in enumerate(state):
                if s:
                    merged[i] = 1
        result[tuple(merged)] += prob
    return _normalize(dict(result))


def mixture(
    distributions: list[Distribution], weights: list[float] | None = None
) -> Distribution:
    """A weighted convex combination of distributions (pick one per draw)."""
    if not distributions:
        raise ValueError("mixture requires at least one distribution")
    _shared_n(distributions)
    if weights is None:
        weights = [1.0] * len(distributions)
    if len(weights) != len(distributions):
        raise ValueError("weights must match the number of distributions")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    result: Distribution = defaultdict(float)
    for dist, weight in zip(distributions, weights, strict=True):
        for state, prob in dist.items():
            result[state] += (weight / total) * prob
    return _normalize(dict(result))


def sample(distribution: Distribution, size: int, *, seed: int) -> list[tuple[int, ...]]:
    """Draw ``size`` i.i.d. states from a distribution (seeded, isolated RNG).

    Uses ``np.random.default_rng(seed)`` — never the global RNG — so a draw is
    reproducible from ``seed`` alone. A convenience for inspecting an
    environment; ``MatchingAnalysis.matching`` does its own seeded sampling.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative; got {size}")
    rng = np.random.default_rng(seed)
    states = list(distribution.keys())
    probs = np.array(list(distribution.values()), dtype=float)
    probs /= probs.sum()
    indices = rng.choice(len(states), size=size, p=probs)
    return [states[i] for i in indices]
