"""Estimate substrates from observed transitions, with epistemic uncertainty.

The TPM that IIT requires is interventional — ``p(next | do(current))``
under a uniform perturbation of all current states — so what data can
legitimately provide depends on how the data was produced. The caller must
assert the ``regime``:

- ``"perturbational"``: each transition is an independent trial in which the
  current state was set by intervention. When the trials cover the state
  space, the estimand is identified and estimation is a counting problem.
- ``"observational"``: transitions come from a passively recorded
  trajectory. Treating them as interventional assumes the recorded dynamics
  are the causal dynamics (no unobserved driver, correct units and grain,
  stationarity) — assumptions about the world that the data cannot verify.
  States the trajectory never visits are *unidentified*, not merely
  unsampled: substrates that differ only on unvisited rows produce
  identical data and can have materially different Φ. The
  :class:`CoverageReport` records exactly which rows the data constrained.

Estimation is per-unit counting with a conjugate symmetric Beta prior on
every cell of the state-by-node TPM (default Jeffreys, ``a = 1/2``). The
result is a :class:`SubstratePosterior` — a distribution over substrates,
never a single point estimate: Φ of a posterior-mean TPM conflates
epistemic uncertainty with genuine indeterminism and suppresses Φ where the
data is merely uninformative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import convert
from .provenance import Provenance
from .substrate import Substrate

REGIMES = ("perturbational", "observational")


def _index_to_state(index: int, n_units: int) -> tuple[int, ...]:
    """Decode a little-endian row index into a binary state tuple."""
    return tuple((index >> i) & 1 for i in range(n_units))


@dataclass(frozen=True, eq=False)
class CoverageReport:
    """Which current states the data constrained, and how often.

    In the perturbational regime, uncovered states mark where more trials
    are needed. In the observational regime they are stronger: the dynamics
    at an unvisited state are unidentified by the data, so any downstream
    quantity that depends on those TPM rows rests on the prior alone.

    Attributes:
        counts: Number of observed transitions out of each current state,
            indexed by little-endian state index; shape ``(2**n_units,)``.
        n_units: Number of units in the substrate.
    """

    counts: NDArray[np.int64]
    n_units: int

    @property
    def n_states(self) -> int:
        """Total number of current states."""
        return 2**self.n_units

    @property
    def uncovered_states(self) -> tuple[tuple[int, ...], ...]:
        """States (as little-endian tuples) with zero observed transitions."""
        return tuple(
            _index_to_state(int(i), self.n_units)
            for i in np.flatnonzero(self.counts == 0)
        )

    @property
    def fraction_covered(self) -> float:
        """Fraction of current states with at least one observation."""
        return float(np.count_nonzero(self.counts)) / self.n_states

    @property
    def is_complete(self) -> bool:
        """Whether every current state was observed at least once."""
        return bool(np.all(self.counts > 0))

    def to_pandas(self) -> pd.DataFrame:
        """Long-format DataFrame with one row per state: ``state``, ``count``."""
        return pd.DataFrame(
            {
                "state": [
                    _index_to_state(i, self.n_units) for i in range(self.n_states)
                ],
                "count": self.counts.astype(int),
            }
        )


@dataclass(frozen=True, eq=False)
class SubstratePosterior:
    """Posterior distribution over substrates given observed transitions.

    Each cell of the state-by-node TPM carries an independent Beta
    posterior: cell ``(s, i)`` is ``Beta(alpha_on[s, i], alpha_off[s, i])``
    over ``p(unit i is ON at t+1 | current state s)``. Rows follow the
    little-endian state order.

    Draw concrete substrates with :meth:`sample`; every existing PyPhi
    computation applies to the samples unchanged.

    Attributes:
        alpha_on: Beta ``a`` parameters, shape ``(2**n, n)``.
        alpha_off: Beta ``b`` parameters, shape ``(2**n, n)``.
        regime: The caller's assertion about how the data was produced
            (``"perturbational"`` or ``"observational"``).
        prior: The symmetric Beta prior pseudocount added to every cell.
        coverage: Per-state observation counts (see :class:`CoverageReport`).
        node_labels: Optional labels passed through to sampled substrates.
        provenance: Environment record captured at estimation time.
    """

    alpha_on: NDArray[np.float64]
    alpha_off: NDArray[np.float64]
    regime: str
    prior: float
    coverage: CoverageReport
    node_labels: Sequence[str] | None
    provenance: Provenance

    @property
    def n_units(self) -> int:
        """Number of units in the substrate."""
        return self.alpha_on.shape[1]

    @property
    def n_states(self) -> int:
        """Total number of current states."""
        return self.alpha_on.shape[0]

    def sample(
        self, *, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> Substrate:
        """Draw one substrate from the posterior.

        Every TPM cell is drawn independently from its Beta posterior;
        exactly one of ``seed`` or ``rng`` must be given.

        Args:
            seed: Seed for a fresh, isolated ``np.random.default_rng``.
            rng: An existing generator to draw from (advances its state).

        Returns:
            An ordinary :class:`~pyphi.substrate.Substrate`.
        """
        if (seed is None) == (rng is None):
            raise ValueError("Provide exactly one of seed= or rng=.")
        if rng is None:
            rng = np.random.default_rng(seed)
        p_on = rng.beta(self.alpha_on, self.alpha_off)
        return Substrate(
            tpm=convert.to_multidimensional(p_on), node_labels=self.node_labels
        )


def estimate_substrate(
    data: NDArray[np.integer] | tuple[NDArray[np.integer], NDArray[np.integer]],
    *,
    regime: str,
    prior: float = 0.5,
    node_labels: Sequence[str] | None = None,
    model: str = "counts",
) -> SubstratePosterior:
    """Estimate a posterior over substrates from binary transition data.

    Args:
        data: Either a pair ``(current, next)`` of integer arrays, each of
            shape ``(T, n)`` with one transition per row (the natural form
            for perturbational trials), or a single integer array of shape
            ``(T, n)`` — a trajectory whose transitions are the consecutive
            row pairs (the natural form for observational recordings). Both
            forms are accepted under both regimes; values must be 0/1.

    Keyword Args:
        regime: Required assertion about how the data was produced:
            ``"perturbational"`` (current states set by intervention) or
            ``"observational"`` (passively recorded trajectory; see the
            module docstring for the identifiability caveat).
        prior: Symmetric Beta prior pseudocount added to every TPM cell
            (default Jeffreys, ``1/2``). Must be positive.
        node_labels: Optional labels for the substrate units.
        model: Estimation model; only ``"counts"`` is implemented.

    Returns:
        A :class:`SubstratePosterior` with independent Beta posteriors over
        every TPM cell and a :class:`CoverageReport`.
    """
    if model != "counts":
        raise NotImplementedError(f"model={model!r}; only 'counts' is implemented")
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if prior <= 0:
        raise ValueError(f"prior must be positive, got {prior}")

    if isinstance(data, np.ndarray):
        trajectory = np.asarray(data)
        current, next_ = trajectory[:-1], trajectory[1:]
    else:
        current, next_ = (np.asarray(a) for a in data)
    if current.ndim != 2 or current.shape != next_.shape:
        raise ValueError(
            "data must be a (T, n) trajectory or a pair of (T, n) arrays "
            f"of equal shape; got shapes {current.shape} and {next_.shape}"
        )
    if not (np.isin(current, (0, 1)).all() and np.isin(next_, (0, 1)).all()):
        raise ValueError("estimation is binary-only; data values must be 0 or 1")

    n = current.shape[1]
    # Little-endian row index of each current state.
    rows = current.astype(np.int64) @ (1 << np.arange(n))
    counts_on = np.zeros((2**n, n))
    counts_off = np.zeros((2**n, n))
    np.add.at(counts_on, rows, next_)
    np.add.at(counts_off, rows, 1 - next_)
    row_counts = np.bincount(rows, minlength=2**n)

    return SubstratePosterior(
        alpha_on=counts_on + prior,
        alpha_off=counts_off + prior,
        regime=regime,
        prior=prior,
        coverage=CoverageReport(counts=row_counts, n_units=n),
        node_labels=node_labels,
        provenance=Provenance.capture(),
    )
