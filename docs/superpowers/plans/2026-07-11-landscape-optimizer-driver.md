# Landscape Optimizer Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pyphi.optimize` — a population-first black-box optimizer that searches over a substrate's connection weights for a substrate maximizing an IIT quantity (default: signed normalized φₛ).

**Architecture:** A single new module `pyphi/optimize.py`, sibling to `pyphi/landscape.py`. `optimize()` poses a bounded minimization to `scipy.optimize.differential_evolution` (vectorized, unpolished, seeded); the vectorized objective builds one substrate per population member, evaluates all of them through the public `analyze(..., compute="sia")` — sequentially or fanned out through the existing `map_reduce` — reads the chosen scalar off each SIA, and logs every evaluation to a trajectory DataFrame. Returns a frozen `OptimizationResult`.

**Tech Stack:** Python 3.13+, NumPy, pandas, `scipy.optimize.differential_evolution` (already a dependency, v1.17), `pyphi.analyze`, `pyphi.parallel.map_reduce`, `pyphi.substrate_generator.build_substrate`.

## Global Constraints

- Python 3.13+ only; no backward-compatibility shims.
- No new runtime dependency: use `scipy.optimize.differential_evolution` (installed, v1.17), behind an internal seam so `cma` can slot in later.
- Reproducibility: `optimize()` takes a **required** `seed` argument, drives the optimizer from an isolated `np.random.default_rng(seed)`, and records the seed on the result. No module-level `np.random.seed`.
- Raw data: the returned `OptimizationResult.trajectory` holds one row per evaluation (parameters + objective + reachability + selection identity + margins); `.save(path)` serializes it. The library never writes files on its own; no-clobber filenames are the caller's job.
- Docstrings: NumPy style (underlined `Parameters`/`Returns`/`Raises`/`Notes`), final-state impersonal voice, Unicode symbols (`φₛ`, `θ`), doctests executable. No planning-artifact or migration narrative in source.
- Cite the gate: the driver's objective default and population-first choice trace to `experiments/substrate_landscape_experiments/FINDINGS.md`.
- The driver reads only the scalar objective to decide a move — never a MIP/selection identity (the gate's stability caveat).
- scipy 1.17: pass the RNG via `rng=` (not the deprecated `seed=`); set `vectorized=True`, `updating="deferred"`, `polish=False`, `workers=1`.
- The pytest config sets `filterwarnings = ["error", …]` — any un-ignored warning fails the test. Emit none: `updating="deferred"` preempts scipy's vectorized-override `UserWarning`.

---

## File Structure

- **Create `pyphi/optimize.py`** — `weight_axes`, `OptimizationResult`, `optimize`, and the private helpers `_objective_value`, `_eval_batch`, `_run_backend`, `_build_substrates`.
- **Create `test/test_optimize.py`** — unit + integration tests.
- **Modify `pyphi/__init__.py`** — lift `optimize`, `OptimizationResult`, `weight_axes`; add to `__all__`.
- **Create `changelog.d/landscape-optimizer.feature.md`** — one-line user-facing note.
- **Modify `docs/howto/landscape.md`** — add an "Optimizing over weights" section.
- **Modify `ROADMAP.md`** — add a `pyphi.optimize` dashboard row; flip the Wave 7 "Landscape optimizer driver" bullet and the `pyphi.landscape` row's "open Wave-7 build" phrasing to landed.

Reference patterns (read before starting):
- `pyphi/landscape.py` — `weight_axis` (the single-weight axis this generalizes), `_eval_point` (the exact `analyze(..., compute="sia")` call and the SIA attributes read), `LandscapeSection` (the frozen-result + `to_pandas` shape).
- `pyphi/sweep.py:193-230` (`_run_cells_parallel`) — the `map_reduce` + `config.override(**presets.by_name[formalism], parallel=False)` worker pattern this reuses.
- `test/test_landscape.py:1-92` — the ising-substrate fixture style and the proven unreachable construction (`examples.basic_substrate()` at state `(0, 1, 1)`).

---

### Task 1: `weight_axes` — the N-dimensional parameter axis

**Files:**
- Create: `pyphi/optimize.py`
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: `pyphi.substrate_generator.build_substrate(unit_functions, weights, **kwargs)`.
- Produces: `weight_axes(unit_functions, weights, indices, **kwargs) -> Callable[[NDArray], Substrate]`. The builder maps a length-`len(indices)` vector θ to the substrate built from `weights` with `weights[indices[k]] = θ[k]`; the base matrix is copied per call, never mutated.

- [ ] **Step 1: Write the failing test**

```python
# test/test_optimize.py
"""Tests for pyphi.optimize: black-box optimization over substrate weights."""

import json

import numpy as np
import pandas as pd
import pytest

import pyphi
from pyphi import examples
from pyphi.substrate_generator import ising
from pyphi.optimize import OptimizationResult
from pyphi.optimize import optimize
from pyphi.optimize import weight_axes

# The IIT 4.0 (2023) Fig. 1A substrate; STATE is reachable with positive φ_s.
FIG1A_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
STATE = (1, 0, 0)


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


def test_weight_axes_sets_indexed_entries_without_mutating():
    original = FIG1A_WEIGHTS.copy()
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    substrate = axis(np.array([0.55, 0.35]))
    # Base matrix untouched.
    np.testing.assert_array_equal(FIG1A_WEIGHTS, original)
    # The built substrate is a real Substrate carrying the varied weights.
    assert substrate.node_labels is not None
    baseline = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [], temperature=0.25)
    # A different vector yields a different substrate (weights actually applied).
    other = axis(np.array([0.10, 0.90]))
    assert substrate.tpm.tpm.shape == other.tpm.tpm.shape
    assert not np.array_equal(substrate.tpm.tpm, other.tpm.tpm)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py::test_weight_axes_sets_indexed_entries_without_mutating -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.optimize'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/optimize.py
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

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
        entries=[tuple(index) for index in indices],
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
```

`_WeightAxis` needs `from dataclasses import dataclass` at the top of the module (added in Task 2 anyway; if Task 1 lands first, add the import now).

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py::test_weight_axes_sets_indexed_entries_without_mutating -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/optimize.py test/test_optimize.py
git commit -m "Add weight_axes: N-dimensional substrate parameter axis"
```

---

### Task 2: `OptimizationResult` and the objective/batch helpers

**Files:**
- Modify: `pyphi/optimize.py`
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: `pyphi.analyze.analyze`, `pyphi.exceptions.StateUnreachable*Error`, `pyphi.direction.Direction`.
- Produces:
  - `OptimizationResult` — frozen dataclass with fields `best_params: NDArray`, `best_objective: float`, `best_substrate`, `best_sia`, `trajectory: pd.DataFrame`, `bounds: list[tuple[float, float]]`, `seed: int`, `direction: str`, `objective_name: str`, `settings: dict`, `config_snapshot: dict`, `n_evaluations: int`, `n_unreachable: int`; methods `to_pandas() -> pd.DataFrame` (returns `trajectory`) and `save(path) -> None`.
  - `_UNREACHABLE_PENALTY: float = 1e18` — the fixed sentinel objective (internal minimization convention: large = bad), plus `_UNREACHABLE` (the tuple of unreachable exceptions) and `_QUANTITIES` (the four φₛ names).
  - `_objective_value(sia, objective) -> float` — the natural (maximization-convention) scalar for one SIA: `getattr(sia, name)` for a named quantity (`None` → `nan`), or `objective(sia)` for a callable.
  - `_eval_one(theta, *, builder, state, subset, formalism, objective) -> dict` — evaluate one candidate (`theta` first-positional, the rest keyword-only, so `functools.partial` can bind them and `map_reduce` pass `theta` positionally). Returns a row dict with keys `objective` (natural convention; `nan` if unreachable), `reachable` (bool), `partition`, `cause_state`, `effect_state`, `partition_margin`, `cause_state_margin`, `effect_state_margin`, and `_sia` (the SIA object or `None`).

- [ ] **Step 1: Write the failing test**

```python
# test/test_optimize.py  (append)
from pyphi.optimize import _eval_one
from pyphi.optimize import _objective_value


def test_objective_value_named_and_callable():
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    sia = pyphi.analyze(axis(np.array([0.7])), STATE, compute="sia")
    assert _objective_value(sia, "signed_normalized_phi") == pytest.approx(
        float(sia.signed_normalized_phi)
    )
    assert _objective_value(sia, lambda s: 2.0 * float(s.phi)) == pytest.approx(
        2.0 * float(sia.phi)
    )


def test_eval_one_reachable_row_carries_margins_and_sia():
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    row = _eval_one(
        np.array([0.7]), builder=axis, state=STATE, subset=None, formalism=None,
        objective="signed_normalized_phi",
    )
    assert row["reachable"] is True
    assert row["_sia"] is not None
    assert np.isfinite(row["objective"])
    assert row["cause_state"] == tuple(row["_sia"].system_state.cause.state)
    assert np.isfinite(row["partition_margin"])


def test_eval_one_unreachable_row_is_penalized_not_raised():
    substrate = examples.basic_substrate()  # deterministic; (0,1,1) never reached

    def build(_theta):
        return substrate

    row = _eval_one(
        np.array([0.0]), builder=build, state=(0, 1, 1), subset=None, formalism=None,
        objective="signed_normalized_phi",
    )
    assert row["reachable"] is False
    assert row["_sia"] is None
    assert np.isnan(row["objective"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -k "objective_value or eval_one" -v`
Expected: FAIL — `ImportError: cannot import name '_eval_one'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/optimize.py  (add imports at top; dataclass already imported in Task 1)
import math
from pathlib import Path

import pandas as pd

from pyphi import exceptions

_UNREACHABLE = (
    exceptions.StateUnreachableForwardsError,
    exceptions.StateUnreachableBackwardsError,
)
_QUANTITIES = frozenset(
    {"phi", "signed_phi", "normalized_phi", "signed_normalized_phi"}
)
_MARGIN_NAMES = ("partition", "cause_state", "effect_state")
# Rank-based DE selection only needs unreachable to compare worse than any real
# candidate; a fixed large sentinel (internal minimization convention) suffices.
_UNREACHABLE_PENALTY = 1e18


def _optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _objective_value(sia: Any, objective: Any) -> float:
    """The natural (maximization-convention) objective scalar for one SIA."""
    if callable(objective):
        return float(objective(sia))
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
        sia = analyze(
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


def _part_id(partition: Any) -> str:
    """A stable, opaque identity string for a system partition (compare, don't parse)."""
    if partition is None or getattr(partition, "is_null", False):
        return "NullCut"
    set_partition = getattr(partition, "set_partition", None)
    if set_partition is not None:
        parts = str(sorted(sorted(part) for part in set_partition))
    else:
        parts = type(partition).__name__
    return f"{parts}|{sorted(partition.removed_edges())}"


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -k "objective_value or eval_one" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/optimize.py test/test_optimize.py
git commit -m "Add OptimizationResult and objective/candidate-evaluation helpers"
```

---

### Task 3: `optimize` — the sequential driver

**Files:**
- Modify: `pyphi/optimize.py`
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: `weight_axes`, `_eval_one`, `OptimizationResult`, `_UNREACHABLE_PENALTY`, `_QUANTITIES`; `scipy.optimize.differential_evolution`; `pyphi.conf.config`, `pyphi.conf.fallback`.
- Produces: `optimize(builder, state, bounds, *, seed, objective="signed_normalized_phi", direction="maximize", x0=None, popsize=15, maxiter=100, tol=0.01, subset=None, formalism=None, parallel=None, progress=None) -> OptimizationResult`. Also the private `_eval_batch(builder, thetas, state, subset, formalism, objective, parallel, progress) -> list[dict]` (sequential in this task; the parallel branch is Task 4).

- [ ] **Step 1: Write the failing test**

```python
# test/test_optimize.py  (append)
def test_optimize_beats_random_baseline():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    bounds = [(0.1, 1.3), (0.1, 1.3)]
    result = optimize(
        axis, STATE, bounds, seed=20260711, popsize=6, maxiter=8, parallel=False
    )
    # A seeded random baseline over the same evaluation budget.
    rng = np.random.default_rng(20260711)
    n = result.n_evaluations
    baseline = max(
        _eval_one(
            rng.uniform([0.1, 0.1], [1.3, 1.3]), builder=axis, state=STATE,
            subset=None, formalism=None, objective="signed_normalized_phi",
        )["objective"]
        for _ in range(n)
    )
    assert result.best_objective >= baseline
    assert result.direction == "maximize"
    assert result.seed == 20260711
    assert len(result.trajectory) == result.n_evaluations
    # The best row's objective matches best_objective.
    assert result.trajectory["objective"].max() == pytest.approx(result.best_objective)


def test_optimize_is_reproducible():
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    bounds = [(0.2, 1.2)]
    kwargs = dict(seed=7, popsize=5, maxiter=5, parallel=False)
    r1 = optimize(axis, STATE, bounds, **kwargs)
    r2 = optimize(axis, STATE, bounds, **kwargs)
    pd.testing.assert_frame_equal(r1.trajectory, r2.trajectory)
    np.testing.assert_array_equal(r1.best_params, r2.best_params)


def test_optimize_logs_unreachable_not_raised():
    substrate = examples.basic_substrate()
    result = optimize(
        lambda _t: substrate, (0, 1, 1), [(0.0, 1.0)],
        seed=1, popsize=4, maxiter=3, parallel=False,
    )
    assert result.n_unreachable == result.n_evaluations
    assert not result.trajectory["reachable"].any()


def test_optimize_rejects_unknown_objective_name():
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    with pytest.raises(ValueError, match="unknown objective"):
        optimize(axis, STATE, [(0.2, 1.2)], seed=1, objective="not_a_quantity")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -k optimize -v`
Expected: FAIL — `ImportError: cannot import name 'optimize'` (the import at the top of the test file already references it; whole file errors at collection until implemented).

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/optimize.py  (add imports)
from pyphi.conf import config
from pyphi.conf import fallback


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
    """Evaluate a whole population. Sequential here; parallel branch in Task 4."""
    return [
        _eval_one(theta, builder=builder, state=state, subset=subset,
                  formalism=formalism, objective=objective)
        for theta in thetas
    ]


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
    popsize, maxiter, tol : int, int, float
        Differential-evolution budget and convergence controls.
    subset : Sequence[int], optional
        Candidate system node indices; ``None`` uses the whole substrate.
    formalism : str, optional
        A formalism preset applied for these evaluations only.
    parallel : bool, optional
        Evaluate each generation's population through ``map_reduce``; defaults to
        ``config.infrastructure.parallel``.
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
    """
    from scipy.optimize import differential_evolution

    if isinstance(objective, str) and objective not in _QUANTITIES:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of "
            f"{', '.join(sorted(_QUANTITIES))}, or a callable"
        )
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")

    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
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
        bounds,
        rng=np.random.default_rng(seed),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        x0=x0,
        vectorized=True,
        updating="deferred",  # required with vectorized; set explicitly so scipy
        # does not emit the override UserWarning (the pytest config escalates
        # warnings to errors).
        polish=False,
        workers=1,
    )

    trajectory = pd.DataFrame(rows)
    # Derive the best from the logged trajectory (not `outcome`) so best_params /
    # best_sia / best_objective stay mutually consistent and match the table.
    # idxmax/idxmin skip NaN (unreachable) rows.
    objectives = trajectory["objective"]
    if objectives.notna().any():
        best_row = int(objectives.idxmax() if direction == "maximize"
                       else objectives.idxmin())
        best_objective = float(objectives.iloc[best_row])
        best_params = np.array(
            [trajectory.iloc[best_row][f"p{d}"] for d in range(len(bounds))],
            dtype=float,
        )
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
        bounds=bounds,
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
        n_unreachable=int((~trajectory["reachable"]).sum()),
    )
```

Note: the best is read back from the logged trajectory, not from `outcome`, so `best_params`, `best_sia`, and `best_objective` are guaranteed mutually consistent and equal to the trajectory's extreme row. The all-unreachable branch avoids `int(NaN)` from `idxmax` on an all-NaN column. For `direction="minimize"`, `idxmin` and the trajectory minimum are used — this is the sign handling Task 5 verifies.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -k optimize -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/optimize.py test/test_optimize.py
git commit -m "Add optimize(): sequential population driver over substrate weights"
```

---

### Task 4: Parallel population evaluation via `map_reduce`

**Files:**
- Modify: `pyphi/optimize.py` (`_eval_batch` parallel branch)
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: `pyphi.parallel.map_reduce`, `pyphi.conf.presets`; the module-level `_eval_one` (picklable) via `functools.partial`.
- Produces: no new public symbol; `_eval_batch` gains a `parallel=True` branch that evaluates the whole population in one `map_reduce` call with inner parallelism disabled.

- [ ] **Step 1: Write the failing test**

```python
# test/test_optimize.py  (append)
def test_parallel_matches_sequential_best():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    bounds = [(0.1, 1.3), (0.1, 1.3)]
    kwargs = dict(seed=99, popsize=5, maxiter=5)
    seq = optimize(axis, STATE, bounds, parallel=False, **kwargs)
    par = optimize(axis, STATE, bounds, parallel=True, **kwargs)
    assert par.best_objective == pytest.approx(seq.best_objective)
    np.testing.assert_allclose(par.best_params, seq.best_params)
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py::test_parallel_matches_sequential_best -v`
Expected: PASS already — Task 3's `_eval_batch` ignores the `parallel` flag and runs sequentially either way, so both runs match trivially. This is a correctness regression guard for the parallel branch, not a failing-first test. After implementing Step 3, confirm the parallel path is actually taken by temporarily putting `raise AssertionError("parallel path")` inside the `map_reduce` branch and re-running with `parallel=True` (it should error), then remove it.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/optimize.py  — replace _eval_batch body
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
            _eval_one(theta, builder=builder, state=state, subset=subset,
                      formalism=formalism, objective=objective)
            for theta in thetas
        ]

    from functools import partial

    from pyphi.conf import presets
    from pyphi.parallel import map_reduce

    preset = presets.by_name[formalism] if formalism is not None else {}
    # partial binds the fixed args as keywords; theta stays the one positional,
    # so cell_fn is a picklable one-argument worker (module-level function +
    # picklable bound args) — hence weight_axes returns a picklable object and
    # not a closure.
    cell_fn = partial(
        _eval_one,
        builder=builder,
        state=state,
        subset=subset,
        formalism=formalism,
        objective=objective,
    )
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
```

**Picklability requirement** (document in `optimize`'s `Notes`): `parallel=True`
sends the `builder` and `objective` to worker processes, so both must be
picklable. `weight_axes` returns a picklable builder and the named string
objectives are picklable, so the default path parallelizes. A closure builder or
a lambda objective must be run with `parallel=False`, or they raise a clear
pickling error from the process backend.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -v`
Expected: PASS (all tests, including the Task 2/Task 3 tests after the `_eval_one` signature refactor).

- [ ] **Step 5: Commit**

```bash
git add pyphi/optimize.py test/test_optimize.py
git commit -m "Evaluate optimizer populations through map_reduce when parallel"
```

---

### Task 5: Callable objective end-to-end, `save`, and `to_pandas`

**Files:**
- Modify: `pyphi/optimize.py` (no new logic expected; verify `direction="minimize"` path)
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: everything above.
- Produces: no new symbol; locks the callable-objective, minimize-direction, and `save`/`to_pandas` behaviors with tests.

- [ ] **Step 1: Write the failing test**

```python
# test/test_optimize.py  (append)
def test_optimize_callable_objective_and_minimize(tmp_path):
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    result = optimize(
        axis, STATE, [(0.2, 1.2)],
        seed=3, objective=lambda sia: float(sia.phi), direction="minimize",
        popsize=5, maxiter=5, parallel=False,
    )
    assert result.objective_name == "<callable>"
    assert result.direction == "minimize"
    # Minimizing φ: the best is the smallest logged objective.
    assert result.best_objective == pytest.approx(result.trajectory["objective"].min())


def test_result_save_and_to_pandas_roundtrip(tmp_path):
    axis = weight_axes([ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25)
    result = optimize(axis, STATE, [(0.2, 1.2)], seed=5, popsize=4, maxiter=3,
                      parallel=False)
    assert result.to_pandas() is result.trajectory
    path = tmp_path / "run_seed5.json"
    result.save(path)
    payload = json.loads(path.read_text())
    assert payload["seed"] == 5
    assert len(payload["trajectory"]) == result.n_evaluations
    assert payload["best_objective"] == pytest.approx(result.best_objective)
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -k "callable or roundtrip" -v`
Expected: PASS if Tasks 2–3 implemented `save`, callable objective, and the minimize sign correctly; if any assertion fails, fix the corresponding logic in `optimize`/`OptimizationResult.save` (this task is the gate that confirms those paths).

- [ ] **Step 3: Fix any gaps surfaced**

The minimize path (`idxmin` on the trajectory) and `save` are already implemented in Tasks 2–3, so these tests should pass as written; this task is the gate that confirms it. If an assertion fails, the fix is a correction to the existing Task 2/3 blocks (e.g. a missing `save` field, or a `direction` sign slip), not new structure — no new code block here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u VIRTUAL_ENV uv run pytest test/test_optimize.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add pyphi/optimize.py test/test_optimize.py
git commit -m "Lock callable-objective, minimize, and result-serialization behavior"
```

---

### Task 6: Package exposure, changelog, docs, and ROADMAP

**Files:**
- Modify: `pyphi/__init__.py`
- Create: `changelog.d/landscape-optimizer.feature.md`
- Modify: `docs/howto/landscape.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Produces: top-level `pyphi.optimize`, `pyphi.OptimizationResult`, `pyphi.weight_axes`.

- [ ] **Step 1: Lift the public surface in `pyphi/__init__.py`**

After the existing landscape imports (`pyphi/__init__.py:107-111`), add:

```python
from .optimize import OptimizationResult as OptimizationResult
from .optimize import optimize as optimize
from .optimize import weight_axes as weight_axes
```

And add `"optimize"`, `"OptimizationResult"`, `"weight_axes"` to the `__all__` list near the existing `"landscape_section"` entry.

- [ ] **Step 2: Verify the top-level surface imports**

Run: `env -u VIRTUAL_ENV uv run python -c "import pyphi; print(pyphi.optimize, pyphi.OptimizationResult, pyphi.weight_axes)"`
Expected: three objects print, no ImportError.

- [ ] **Step 3: Add the changelog fragment**

```bash
printf '`pyphi.optimize()`: a seeded, population-based black-box optimizer that searches over substrate connection weights for a maximizer of signed normalized φₛ (or another IIT quantity), with `weight_axes` building the search space and a saved per-evaluation trajectory.\n' > changelog.d/landscape-optimizer.feature.md
```

- [ ] **Step 4: Add a how-to section to `docs/howto/landscape.md`**

Append a section "Optimizing over weights", matching the file's existing prose
and fenced-code style, with this example:

````markdown
## Optimizing over weights

`landscape_section` shows φₛ *along* an axis; `pyphi.optimize` *searches* over
one. Give it a builder mapping a parameter vector to a substrate (`weight_axes`
builds one), a box of bounds, and a seed:

```python
import numpy as np
import pyphi
from pyphi.substrate_generator import ising

weights = np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]])
axis = pyphi.weight_axes(
    [ising.probability] * 3, weights, [(0, 1), (1, 0)], temperature=0.25
)
result = pyphi.optimize(
    axis, (1, 0, 0), [(0.1, 1.3), (0.1, 1.3)], seed=20260711, popsize=8, maxiter=20
)
print(result.best_objective, result.best_params)
df = result.to_pandas()  # per-evaluation trajectory
```

The objective defaults to signed normalized φₛ, the quantity that stays
continuous across MIP switches. Every evaluation is logged to
`result.to_pandas()`, and `result.save("run_seed20260711.json")` persists it.
```
````

Since `docs/*.md` how-tos are not doctest-collected, verify by reading and by
running the body once manually:

Run: `env -u VIRTUAL_ENV uv run python -c "import numpy as np, pyphi; from pyphi.substrate_generator import ising; w=np.array([[-0.2,0.7,0.2],[0.7,-0.2,0.0],[0.0,-0.8,0.2]]); ax=pyphi.weight_axes([ising.probability]*3,w,[(0,1),(1,0)],temperature=0.25); r=pyphi.optimize(ax,(1,0,0),[(0.1,1.3),(0.1,1.3)],seed=20260711,popsize=6,maxiter=8); print(r.best_objective)"`
Expected: prints a float, no error.

- [ ] **Step 5: Update `ROADMAP.md`**

1. Add a Status Dashboard row after the `pyphi.landscape` row (line ~66):

   `| pyphi.optimize | ✅ landed | — | Wave 7 item 3: population-first black-box optimizer over substrate weights. optimize() runs scipy differential_evolution (vectorized, unpolished, seeded) with the objective defaulting to gate-validated signed normalized φₛ; weight_axes builds the vector search space; populations fan out through map_reduce; a per-evaluation trajectory (params, objective, reachability, selection identity, margins) is returned and saveable. Named φₛ variants plus an opt-in callable escape hatch for unvalidated (CES/Φ) objectives. Gate: experiments/substrate_landscape_experiments/FINDINGS.md. Spec/plan: 2026-07-11-landscape-optimizer-driver-{design,}.md. |`

2. In the Wave 7 section (line ~193), change the "**Landscape optimizer driver** *(open build)*" bullet's status to landed, keeping the gate verdict and the selection-identity caveat, and pointing at `pyphi.optimize`.

3. In the `pyphi.landscape` row (line ~66), change "so the driver is unblocked and ships population-first — now an open Wave-7 build" to note the driver has landed as `pyphi.optimize`.

- [ ] **Step 6: Commit**

```bash
git add pyphi/__init__.py changelog.d/landscape-optimizer.feature.md docs/howto/landscape.md ROADMAP.md
git commit -m "Expose pyphi.optimize; changelog, how-to, and ROADMAP updates"
```

---

## Final Verification

- [ ] **Full suite including doctests** (per project rule: run with no path argument at least once for `pyphi/` source changes):

Run: `env -u VIRTUAL_ENV uv run pytest`
Expected: green, including the new `pyphi/optimize.py` docstrings collected by `--doctest-modules`.

- [ ] **Type check:**

Run: `env -u VIRTUAL_ENV uv run pyright pyphi/optimize.py`
Expected: no new errors.

- [ ] **Lint/format:**

Run: `env -u VIRTUAL_ENV uv run pre-commit run --files pyphi/optimize.py test/test_optimize.py pyphi/__init__.py`
Expected: pass (or auto-format then re-stage; never `--no-verify`).

---

## Self-Review Notes (addressed)

- **Spec coverage:** `weight_axes` (Task 1); objective handling + `OptimizationResult` + unreachable penalty (Task 2); DE backend, seeding, trajectory, direction, x0, bad-name (Task 3); `map_reduce` parallelism (Task 4); callable escape hatch + `save`/`to_pandas` + minimize (Task 5); `__init__` exposure + changelog + how-to + ROADMAP (Task 6). Out-of-scope items (analytic gradient, surrogates, named CES/Φ, checkpointing) intentionally absent.
- **Worktree env:** all `pytest`/`python` commands use `env -u VIRTUAL_ENV uv run …` because the plan executes in `.claude/worktrees/wave7-landscape-optimizer` (memory: worktree venv vs uv mismatch). If `uv run` targets the wrong interpreter, install the worktree venv per the memory recipe first.
- **Picklability (parallel path):** `weight_axes` returns a picklable `_WeightAxis` object (not a closure), `_eval_one` takes `theta` first-positional with the rest keyword-only, and the objective is a picklable string by default — so `partial(_eval_one, ...)` is a picklable one-argument `map_reduce` worker. Closure builders and lambda objectives only work with `parallel=False`; this is documented in `optimize`'s `Notes` and enforced by a clear pickling error otherwise.
- **First-batch penalty:** the unreachable penalty is a fixed sentinel (`_UNREACHABLE_PENALTY`), so an unreachable candidate in DE's very first generation is handled with no "worst-seen-so-far" bootstrapping problem.
