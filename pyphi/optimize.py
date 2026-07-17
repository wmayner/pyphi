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
from pyphi.conf import config
from pyphi.conf import fallback

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
    try:
        value = getattr(sia, objective)
    except AttributeError:
        raise ValueError(
            f"objective {objective!r} is not available on "
            f"{type(sia).__name__}; choose an objective the requested "
            f"formalism provides (e.g. objective='phi')"
        ) from None
    return _optional_float(value)


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
    # Selection margins and state specifications exist only on IIT 4.0
    # SIAs; other formalisms' rows carry None/NaN in those columns.
    system_state = getattr(sia, "system_state", None)
    cause = system_state.cause if system_state is not None else None
    effect = system_state.effect if system_state is not None else None
    margins = getattr(sia, "state_margins", None)
    return {
        "objective": _objective_value(sia, objective),
        "reachable": True,
        "partition": _part_id(sia.partition),
        "cause_state": None if cause is None else tuple(int(x) for x in cause.state),
        "effect_state": None if effect is None else tuple(int(x) for x in effect.state),
        "partition_margin": _optional_float(getattr(sia, "partition_margin", None)),
        "cause_state_margin": _optional_float(
            margins[Direction.CAUSE] if margins is not None else None
        ),
        "effect_state_margin": _optional_float(
            margins[Direction.EFFECT] if margins is not None else None
        ),
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


def _eval_batch(
    builder: Callable[[NDArray[Any]], Any],
    thetas: list[NDArray[Any]],
    state: tuple[int, ...],
    subset: Sequence[int] | None,
    formalism: str | None,
    objective: Any,
    parallel: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    """Evaluate a whole population, sequentially or through ``map_reduce``.

    The parallel branch mirrors ``pyphi.sweep._run_cells_parallel``: one level
    of parallelism (``parallel=True`` on the outer ``map_reduce``, inner
    ``parallel=False`` in the worker config snapshot), one candidate per cell.
    """
    if not parallel or len(thetas) <= 1:
        return [
            _eval_one(
                theta,
                builder=builder,
                state=state,
                subset=subset,
                formalism=formalism,
                objective=objective,
            )
            for theta in thetas
        ]

    from functools import partial

    from pyphi.conf import presets
    from pyphi.parallel import map_reduce

    preset = presets.by_name[formalism] if formalism is not None else {}
    # partial binds the fixed args as keywords; theta stays the one positional,
    # so cell_fn is a picklable one-argument worker.
    cell_fn = partial(
        _eval_one,
        builder=builder,
        state=state,
        subset=subset,
        formalism=formalism,
        objective=objective,
    )
    results: list[Any] = []
    with config.override(parallel=False, **preset):
        results = map_reduce(
            cell_fn,
            thetas,
            parallel=True,
            ordered=True,
            reduce_func=list,
            progress=progress,
            desc="optimize population",
            chunksize=1,
        )
    if len(results) != len(thetas):
        raise AssertionError("map_reduce flattened population results")
    return results


def optimize(
    builder: Callable[[NDArray[Any]], Any],
    state: tuple[int, ...],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int,
    objective: Any = "signed_normalized_phi",
    direction: str = "maximize",
    x0: NDArray[Any] | None = None,
    popsize: int = 15,
    maxiter: int = 100,
    tol: float = 0.01,
    subset: Sequence[int] | None = None,
    formalism: str | None = None,
    parallel: bool | None = None,
    progress: bool | None = None,
) -> OptimizationResult:
    """Search a substrate's weights for a maximizer of an IIT quantity.

    Runs a seeded, population-based black-box optimizer
    (:func:`scipy.optimize.differential_evolution`) over a bounded box of
    parameters, evaluating each candidate as one system irreducibility
    analysis. The default objective, signed normalized φₛ, is continuous across
    minimum-information-partition switches, so the gradient-free search sees no
    discontinuities (``experiments/substrate_landscape_experiments/FINDINGS.md``).

    Parameters
    ----------
    builder : Callable[[NDArray], Substrate]
        The parameter axis: a vector → substrate map. :func:`weight_axes` builds
        one for the single-substrate weight case.
    state : tuple[int, ...]
        The substrate state analyzed at every candidate.
    bounds : Sequence[tuple[float, float]]
        ``(low, high)`` per parameter dimension. ``len(bounds)`` fixes the
        search dimension.
    seed : int
        Seeds an isolated ``np.random.default_rng`` driving the optimizer, and
        is recorded on the result. Required — reproducibility is not optional.
    objective : str or Callable
        ``"signed_normalized_phi"`` (default), ``"phi"``, ``"signed_phi"``, or
        ``"normalized_phi"`` by name, or a callable ``SIA → float`` for
        objectives the roughness gate did not validate.
    direction : str
        ``"maximize"`` (default) or ``"minimize"``.
    x0 : NDArray, optional
        Seeds one member of the initial population, e.g. a known-good
        substrate's weights.
    popsize : int
        Differential-evolution population multiplier.
    maxiter : int
        Maximum number of generations.
    tol : float
        Relative convergence tolerance.
    subset : Sequence[int], optional
        Candidate system node indices; ``None`` uses the whole substrate.
    formalism : str, optional
        A formalism preset applied for these evaluations only.
    parallel : bool, optional
        Evaluate each generation's population through ``map_reduce``; defaults to
        ``config.infrastructure.parallel``. Requires ``builder`` and a callable
        ``objective`` to be picklable (:func:`weight_axes` and the named string
        objectives are).
    progress : bool, optional
        Show a progress bar; defaults to ``config.infrastructure.progress_bars``.

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ValueError
        If ``objective`` is a name that is not one of the four φₛ variants, or
        ``direction`` is neither ``"maximize"`` nor ``"minimize"``.

    Notes
    -----
    The driver reads only the scalar objective to choose a move; it never
    branches on a selection identity, which flips between near-tied selections
    at single-grid-point frequency even where the objective is smooth (the gate
    caveat). Selection identities and margins are logged for inspection only.

    With ``parallel=True`` the ``builder`` and ``objective`` are sent to worker
    processes, so both must be picklable. A closure builder or a lambda
    objective must be run with ``parallel=False``.
    """
    from scipy.optimize import differential_evolution

    if isinstance(objective, str) and objective not in _QUANTITIES:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of "
            f"{', '.join(sorted(_QUANTITIES))}, or a callable"
        )
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"direction must be 'maximize' or 'minimize', got {direction!r}"
        )

    box = [(float(lo), float(hi)) for lo, hi in bounds]
    sign = -1.0 if direction == "maximize" else 1.0  # DE minimizes sign*objective
    show = fallback(progress, config.infrastructure.progress_bars)
    use_parallel = fallback(parallel, config.infrastructure.parallel)
    objective_name = objective if isinstance(objective, str) else "<callable>"

    rows: list[dict[str, Any]] = []
    sias: list[Any] = []
    generation = 0

    def batch_objective(population: NDArray[Any]) -> NDArray[Any]:
        nonlocal generation
        # scipy hands a (D, S) matrix under vectorized=True.
        thetas = [population[:, i] for i in range(population.shape[1])]
        evaluated = _eval_batch(
            builder, thetas, state, subset, formalism, objective, use_parallel, show
        )
        scores = np.empty(len(evaluated))
        for member, (theta, row) in enumerate(zip(thetas, evaluated, strict=True)):
            record = {
                "eval": len(rows),
                "generation": generation,
                **{f"p{d}": float(theta[d]) for d in range(len(theta))},
                "objective": row["objective"],
                "reachable": row["reachable"],
                "partition": row["partition"],
                "cause_state": row["cause_state"],
                "effect_state": row["effect_state"],
                "partition_margin": row["partition_margin"],
                "cause_state_margin": row["cause_state_margin"],
                "effect_state_margin": row["effect_state_margin"],
            }
            rows.append(record)
            sias.append(row["_sia"])
            if row["reachable"]:
                scores[member] = sign * row["objective"]
            else:
                scores[member] = _UNREACHABLE_PENALTY
        generation += 1
        return scores

    outcome = differential_evolution(
        batch_objective,
        box,
        rng=np.random.default_rng(seed),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        x0=x0,
        vectorized=True,
        updating="deferred",
        polish=False,
        workers=1,
    )

    trajectory = pd.DataFrame(rows)
    # Derive the best from the logged trajectory (not `outcome`) so best_params /
    # best_sia / best_objective stay mutually consistent and match the table.
    # nanargmax/nanargmin skip NaN (unreachable) rows; the default RangeIndex
    # makes the returned position line up with the `sias` list.
    values = trajectory["objective"].to_numpy(dtype=float)
    param_cols = [f"p{d}" for d in range(len(box))]
    if np.isfinite(values).any():
        best_row = int(
            np.nanargmax(values) if direction == "maximize" else np.nanargmin(values)
        )
        best_objective = float(values[best_row])
        best_params = trajectory.iloc[best_row][param_cols].to_numpy(dtype=float)
        best_sia = sias[best_row]
    else:
        # Every candidate was dynamically unreachable — no meaningful optimum.
        best_objective = math.nan
        best_params = np.asarray(outcome.x, dtype=float)
        best_sia = None
    best_substrate = builder(best_params)

    return OptimizationResult(
        best_params=best_params,
        best_objective=float(best_objective),
        best_substrate=best_substrate,
        best_sia=best_sia,
        trajectory=trajectory,
        bounds=box,
        seed=int(seed),
        direction=direction,
        objective_name=objective_name,
        settings={
            "backend": "scipy.differential_evolution",
            "popsize": popsize,
            "maxiter": maxiter,
            "tol": tol,
        },
        config_snapshot={
            "precision": config.numerics.precision,
            "formalism": formalism,
        },
        n_evaluations=len(trajectory),
        n_unreachable=int(np.count_nonzero(~trajectory["reachable"].to_numpy())),
    )
