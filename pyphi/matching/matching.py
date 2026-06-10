"""Matching: the expected differentiation gap between world and noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyphi import utils

from .differentiation import Differentiation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .perception import Perception


@dataclass(frozen=True)
class MatchingResult:
    """The result of a matching computation (Eq 21).

    Carries everything needed to reproduce the value: the ``seed``, the
    sampling parameters, and the per-trial perceptual differentiation of the
    world and noise sequences. ``value`` equals the mean of
    ``world_differentiation`` minus the mean of ``noise_differentiation``.
    ``subsequence`` is the winning 1-based inclusive window ``(a, b)`` when
    the maximum was taken over contiguous subsequences (``None`` when the
    full sequence was used); the per-trial values are those of that window.
    """

    value: float
    seed: int
    n_trials: int
    k: int
    world_differentiation: tuple[float, ...]
    noise_differentiation: tuple[float, ...]
    subsequence: tuple[int, int] | None = None


@dataclass(frozen=True)
class MatchingAnalysis:
    """Matching between a system and a world distribution over stimuli.

    ``perceptions`` maps each stimulus to the perceptual structure it
    triggers; ``world_distribution`` gives the probability of each stimulus
    in the world. Every stimulus the world can produce must have a
    perceptual structure; the noise distribution is uniform over all stimuli
    that have one.
    """

    perceptions: Mapping[tuple[int, ...], Perception]
    world_distribution: Mapping[tuple[int, ...], float]

    def __post_init__(self):
        if not self.perceptions:
            raise ValueError("perceptions must contain at least one stimulus")
        missing = set(self.world_distribution) - set(self.perceptions)
        if missing:
            raise ValueError(
                f"world stimuli without a perceptual structure: {sorted(missing)}"
            )
        probabilities = list(self.world_distribution.values())
        if any(p < 0 for p in probabilities):
            raise ValueError("world probabilities must be nonnegative")
        total = float(sum(probabilities))
        if not utils.eq(total, 1.0):
            raise ValueError(f"world probabilities must sum to 1 (got {total})")

    @property
    def noise_distribution(self) -> dict[tuple[int, ...], float]:
        """The structureless world: uniform over stimuli with a structure."""
        stimuli = sorted(self.perceptions)
        return {stimulus: 1.0 / len(stimuli) for stimulus in stimuli}

    def matching(
        self,
        *,
        seed: int,
        n_trials: int,
        k: int,
        subsequence_max: bool = False,
    ) -> MatchingResult:
        """Estimate matching M (Eq 21) by seeded Monte Carlo sampling.

        Each trial samples a length-``k`` world sequence (i.i.d. from
        ``world_distribution``) and a length-``k`` noise sequence (i.i.d.
        from ``noise_distribution``) and compares their perceptual
        differentiation. The two sequences are drawn from common random
        numbers (the same uniform deviates mapped through each
        distribution's inverse CDF), so the comparison is paired: identical
        distributions yield identical sequences and a gap of exactly zero.

        By default the gap is computed over the full sequence (Eq 21 with
        ``(a, b) = (1, k)``). With ``subsequence_max=True`` the trial-mean
        gap is computed for every contiguous window ``(a, b)`` and the
        maximum is returned, recording the winning window.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if k < 1:
            raise ValueError("k must be at least 1")
        rng = np.random.default_rng(seed)
        stimuli = sorted(self.perceptions)
        world_cdf = np.cumsum([self.world_distribution.get(s, 0.0) for s in stimuli])
        noise_cdf = np.cumsum(list(self.noise_distribution.values()))
        # Guard against round-off in the final bin; the totals were validated.
        world_cdf[-1] = noise_cdf[-1] = 1.0
        deviates = rng.random((n_trials, k))
        world_sequences = np.searchsorted(world_cdf, deviates, side="right")
        noise_sequences = np.searchsorted(noise_cdf, deviates, side="right")

        # D_p of a sequence depends only on its set of distinct stimuli.
        cache: dict[frozenset[int], float] = {}

        def differentiation_of(indices) -> float:
            key = frozenset(int(i) for i in indices)
            if key not in cache:
                cache[key] = Differentiation(
                    tuple(self.perceptions[stimuli[i]] for i in sorted(key))
                ).perceptual_differentiation
            return cache[key]

        def per_trial(a: int, b: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
            world = tuple(
                differentiation_of(world_sequences[t, a - 1 : b])
                for t in range(n_trials)
            )
            noise = tuple(
                differentiation_of(noise_sequences[t, a - 1 : b])
                for t in range(n_trials)
            )
            return world, noise

        def gap(world, noise) -> float:
            return float(np.mean(np.asarray(world) - np.asarray(noise)))

        subsequence = None
        if subsequence_max:
            windows = {
                (a, b): per_trial(a, b) for a in range(1, k + 1) for b in range(a, k + 1)
            }
            subsequence = max(windows, key=lambda window: gap(*windows[window]))
            world, noise = windows[subsequence]
        else:
            world, noise = per_trial(1, k)

        return MatchingResult(
            value=gap(world, noise),
            seed=seed,
            n_trials=n_trials,
            k=k,
            world_differentiation=world,
            noise_differentiation=noise,
            subsequence=subsequence,
        )
