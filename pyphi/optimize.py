"""Black-box optimization of IIT quantities over substrate parameters.

Where :mod:`pyphi.landscape` analyzes the φ landscape along one axis, this
module searches it: :func:`optimize` runs a seeded population method over a
bounded box of connection weights, looking for a substrate that maximizes an
IIT quantity — by default the signed normalized system irreducibility φₛ,
which stays continuous across minimum-information-partition switches and so
gives a gradient-free search no discontinuities to trip on
(``experiments/substrate_landscape_experiments/FINDINGS.md``).

:func:`weight_axes` builds the search space for the common case: a map from a
parameter vector to a :func:`~pyphi.substrate_generator.build_substrate`
substrate, varying a chosen set of weight-matrix entries.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyphi import exceptions

_UNREACHABLE = (
    exceptions.StateUnreachableForwardsError,
    exceptions.StateUnreachableBackwardsError,
)
_QUANTITIES = frozenset({"phi", "signed_phi", "normalized_phi", "signed_normalized_phi"})
# Rank-based DE selection only needs unreachable to compare worse than any real
# candidate; a fixed large sentinel (internal minimization convention) suffices.
_UNREACHABLE_PENALTY = 1e18


def weight_axes(
    unit_functions: Any,
    weights: NDArray[Any],
    indices: Sequence[tuple[int, int]],
    **kwargs: Any,
) -> Callable[[NDArray[Any]], Any]:
    """Return a parameter axis varying several weights of a generated substrate.

    The vector analogue of :func:`pyphi.landscape.weight_axis`.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable
        Unit function(s), as accepted by
        :func:`pyphi.substrate_generator.build_substrate`.
    weights : ArrayLike
        The base weight matrix; ``weights[i, j]`` is the connection from unit
        ``i`` to unit ``j``. Copied on every call — never mutated.
    indices : Sequence[tuple[int, int]]
        The ``(i, j)`` entries to vary, in the order the parameter vector
        indexes them.

    Returns
    -------
    _WeightAxis
        A picklable callable mapping a length-``len(indices)`` vector θ to the
        substrate built from the weight matrix with ``weights[indices[k]] =
        θ[k]``. Additional keyword arguments forward to ``build_substrate`` on
        every call.

    Notes
    -----
    The returned axis is a picklable object, not a closure, so a population can
    be evaluated across worker processes (``optimize(..., parallel=True)``).

    Setting a weight to exactly 0 removes the connection from the derived
    connectivity matrix (``cm = weights != 0``), a discrete topology change,
    exactly as for :func:`pyphi.landscape.weight_axis`.
    """
    return _WeightAxis(
        unit_functions=unit_functions,
        base=np.array(weights, dtype=float),
        entries=[(int(i), int(j)) for i, j in indices],
        kwargs=dict(kwargs),
    )


@dataclass(frozen=True)
class _WeightAxis:
    """A picklable vector → substrate map varying a fixed set of weight entries.

    A module-level callable rather than a closure so the process backend can
    pickle it when a population is evaluated in parallel.
    """

    unit_functions: Any
    base: NDArray[Any]
    entries: list[tuple[int, int]]
    kwargs: dict[str, Any]

    def __call__(self, theta: NDArray[Any]) -> Any:
        from pyphi.substrate_generator import build_substrate

        varied = self.base.copy()
        for (i, j), value in zip(self.entries, np.asarray(theta), strict=True):
            varied[i, j] = value
        return build_substrate(self.unit_functions, varied, **self.kwargs)


def _optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _part_id(partition: Any) -> str:
    """A stable, opaque identity string for a system partition.

    Compare identities for equality; do not parse them.
    """
    if partition is None or getattr(partition, "is_null", False):
        return "NullCut"
    set_partition = getattr(partition, "set_partition", None)
    if set_partition is not None:
        parts = str(sorted(sorted(part) for part in set_partition))
    else:
        parts = type(partition).__name__
    return f"{parts}|{sorted(partition.removed_edges())}"


def _objective_value(sia: Any, objective: Any) -> float:
    """The natural (maximization-convention) objective scalar for one SIA."""
    if callable(objective):
        value: Any = objective(sia)
        return float(value)
    return _optional_float(getattr(sia, objective))


def _eval_one(
    theta: NDArray[Any],
    *,
    builder: Callable[[NDArray[Any]], Any],
    state: tuple[int, ...],
    subset: Sequence[int] | None,
    formalism: str | None,
    objective: Any,
) -> dict[str, Any]:
    """Evaluate one candidate; unreachable states are penalized, not raised.

    ``theta`` is first-positional and the rest keyword-only so a
    ``functools.partial`` over the fixed arguments is a picklable one-argument
    worker for ``map_reduce``.
    """
    from pyphi.analyze import analyze
    from pyphi.direction import Direction

    try:
        sia: Any = analyze(
            builder(np.asarray(theta)),
            state,
            subset=subset,
            formalism=formalism,
            compute="sia",
        )
    except _UNREACHABLE:
        return {
            "objective": math.nan,
            "reachable": False,
            "partition": None,
            "cause_state": None,
            "effect_state": None,
            "partition_margin": math.nan,
            "cause_state_margin": math.nan,
            "effect_state_margin": math.nan,
            "_sia": None,
        }
    system_state = sia.system_state
    cause = system_state.cause if system_state is not None else None
    effect = system_state.effect if system_state is not None else None
    margins = sia.state_margins
    return {
        "objective": _objective_value(sia, objective),
        "reachable": True,
        "partition": _part_id(sia.partition),
        "cause_state": None if cause is None else tuple(int(x) for x in cause.state),
        "effect_state": None if effect is None else tuple(int(x) for x in effect.state),
        "partition_margin": _optional_float(sia.partition_margin),
        "cause_state_margin": _optional_float(margins[Direction.CAUSE]),
        "effect_state_margin": _optional_float(margins[Direction.EFFECT]),
        "_sia": sia,
    }


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of an :func:`optimize` run.

    Attributes
    ----------
    best_params : NDArray
        The winning parameter vector.
    best_objective : float
        Its objective value, in the chosen-``direction`` convention.
    best_substrate : Substrate
        ``builder(best_params)``.
    best_sia : SystemIrreducibilityAnalysis
        The analysis at the best point — its selection identity and margins say
        how robust the winning substrate is.
    trajectory : pd.DataFrame
        One row per evaluation: ``eval``, ``generation``, one column per
        parameter dimension (``p0``, ``p1``, …), ``objective`` (chosen-direction
        convention), ``reachable``, the selection identity (``partition``,
        ``cause_state``, ``effect_state``) and the selection margins
        (``partition_margin``, ``cause_state_margin``, ``effect_state_margin``).
    bounds : list[tuple[float, float]]
        The search box.
    seed : int
        The seed driving the run.
    direction : str
        ``"maximize"`` or ``"minimize"``.
    objective_name : str
        The quantity name, or ``"<callable>"``.
    settings : dict
        Backend and budget: ``backend``, ``popsize``, ``maxiter``, ``tol``.
    config_snapshot : dict
        PyPhi configuration at run time (``precision``, ``formalism``).
    n_evaluations : int
        Total candidate evaluations.
    n_unreachable : int
        How many candidates had a dynamically unreachable state.
    """

    best_params: NDArray[Any]
    best_objective: float
    best_substrate: Any
    best_sia: Any
    trajectory: pd.DataFrame
    bounds: list[tuple[float, float]]
    seed: int
    direction: str
    objective_name: str
    settings: dict[str, Any]
    config_snapshot: dict[str, Any]
    n_evaluations: int
    n_unreachable: int

    def to_pandas(self) -> pd.DataFrame:
        return self.trajectory

    def save(self, path: Any) -> None:
        """Write the trajectory and metadata to ``path`` as JSON.

        Writes exactly where told; parameter-encoded, non-clobbering filenames
        are the caller's responsibility.
        """
        import json

        payload = {
            "best_params": list(map(float, self.best_params)),
            "best_objective": float(self.best_objective),
            "bounds": [list(map(float, b)) for b in self.bounds],
            "seed": int(self.seed),
            "direction": self.direction,
            "objective_name": self.objective_name,
            "settings": self.settings,
            "config_snapshot": self.config_snapshot,
            "n_evaluations": int(self.n_evaluations),
            "n_unreachable": int(self.n_unreachable),
            "trajectory": self.trajectory.to_dict(orient="records"),
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=str))
