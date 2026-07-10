"""Continuous-parameter analysis of IIT quantities over substrate space.

Every quantity PyPhi computes is a function of the substrate's parameters
(connection weights, TPM entries). This module evaluates those functions
along one parameter axis: :func:`landscape_section` computes a system
irreducibility analysis at every point of a 1-D grid and returns a tidy
table that tracks not only φ but the identity of every discrete selection
behind it (MIP partition, specified cause and effect states) and the
selection margins; :func:`perturb` estimates local derivatives at a single
point by central finite differences.

The φ landscape is piecewise-smooth: within a region where every selection
picks the same winner (a *selection regime*), φ is a smooth function of the
parameters, but the reported φₛ can jump where the MIP or a specified state
switches. The ``regime`` column and :attr:`LandscapeSection.boundaries`
locate those switches; :attr:`Perturbation.same_regime` flags a derivative
estimate that straddles one. Because the raw φₛ jumps at MIP switches while
the normalized value stays continuous, ``signed_phi`` and
``signed_normalized_phi`` are the better-behaved quantities for numerical
work — the positive-part clamp makes ``phi`` exactly flat wherever the raw
integration is negative.

A parameter axis is any callable mapping a float to a
:class:`~pyphi.substrate.Substrate`; :func:`weight_axis` builds one for the
common case of varying a single connection weight of a
:func:`~pyphi.substrate_generator.build_substrate` substrate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pyphi import exceptions
from pyphi.conf import config
from pyphi.conf import fallback

_UNREACHABLE = (
    exceptions.StateUnreachableForwardsError,
    exceptions.StateUnreachableBackwardsError,
)

_QUANTITIES = frozenset({"phi", "signed_phi", "normalized_phi", "signed_normalized_phi"})

_MARGIN_NAMES = ("partition", "cause_state", "effect_state")


def weight_axis(
    unit_functions: Any,
    weights: NDArray[Any],
    index: tuple[int, int],
    **kwargs: Any,
) -> Callable[[float], Any]:
    """Return a parameter axis varying one weight of a generated substrate.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable
        Unit function(s), as accepted by
        :func:`pyphi.substrate_generator.build_substrate`.
    weights : ArrayLike
        The weight matrix; ``weights[i, j]`` is the connection from unit
        ``i`` to unit ``j``. Copied on every call — the input is never
        mutated.
    index : tuple[int, int]
        The ``(i, j)`` entry to vary.

    Returns
    -------
    Callable[[float], Substrate]
        A builder mapping a parameter value θ to the substrate built from
        the weight matrix with ``weights[index] = θ``. Additional keyword
        arguments are forwarded to ``build_substrate`` on every call.

    Notes
    -----
    Setting the weight to exactly 0 removes the connection from the derived
    connectivity matrix (``cm = weights != 0``), a discrete topology change:
    the analysis at θ = 0 may short-circuit as reducible even where the
    values at θ = ±ε do not.
    """
    from pyphi.substrate_generator import build_substrate

    base = np.array(weights, dtype=float)

    def build(theta: float) -> Any:
        varied = base.copy()
        varied[index] = theta
        return build_substrate(unit_functions, varied, **kwargs)

    return build


def _part_id(partition: Any) -> str:
    """A stable, human-readable identity string for a system partition.

    Opaque: compare identities for equality; do not parse them.
    """
    if partition is None or getattr(partition, "is_null", False):
        return "NullCut"
    set_partition = getattr(partition, "set_partition", None)
    if set_partition is not None:
        parts = str(sorted(sorted(part) for part in set_partition))
    else:
        parts = type(partition).__name__
    return f"{parts}|{sorted(partition.removed_edges())}"


def _optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _eval_point(
    builder: Callable[[float], Any],
    theta: float,
    state: tuple[int, ...],
    subset: Sequence[int] | None,
    formalism: str | None,
) -> tuple[dict[str, Any], Any]:
    """Evaluate one SIA at ``theta`` and extract its row."""
    from pyphi.analyze import analyze

    sia: Any = analyze(
        builder(float(theta)), state, subset=subset, formalism=formalism, compute="sia"
    )
    system_state = sia.system_state
    cause = system_state.cause if system_state is not None else None
    effect = system_state.effect if system_state is not None else None
    state_margins = sia.state_margins
    from pyphi.direction import Direction

    row = {
        "phi": float(sia.phi),
        "signed_phi": _optional_float(sia.signed_phi),
        "normalized_phi": _optional_float(sia.normalized_phi),
        "signed_normalized_phi": _optional_float(sia.signed_normalized_phi),
        "partition": _part_id(sia.partition),
        "cause_state": None if cause is None else tuple(int(x) for x in cause.state),
        "effect_state": None if effect is None else tuple(int(x) for x in effect.state),
        "partition_margin": _optional_float(sia.partition_margin),
        "cause_state_margin": _optional_float(state_margins[Direction.CAUSE]),
        "effect_state_margin": _optional_float(state_margins[Direction.EFFECT]),
        "effectively_tied": sia.effectively_tied,
    }
    return row, sia


def _selection_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["partition"], row["cause_state"], row["effect_state"])


@dataclass(frozen=True)
class LandscapeSection:
    """A 1-D section of the φ landscape along one parameter axis.

    Attributes
    ----------
    df : pd.DataFrame
        One row per evaluated grid point, indexed by ``theta``. Columns:
        the four φ variants (``phi``, ``signed_phi``, ``normalized_phi``,
        ``signed_normalized_phi``), the selection identity (``partition``,
        ``cause_state``, ``effect_state``), the selection margins
        (``partition_margin``, ``cause_state_margin``,
        ``effect_state_margin``; ``NaN`` where undefined),
        ``effectively_tied``, and ``regime`` — an integer that increments
        whenever the selection identity differs from the previous point.
    sias : list
        The raw :class:`SystemIrreducibilityAnalysis` objects, aligned 1:1
        with the rows of ``df``.
    skipped : list[float]
        Grid points whose state is dynamically unreachable for the
        substrate built there (no defined repertoire, so no analysis).
    """

    df: pd.DataFrame
    sias: list[Any]
    skipped: list[float]

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    @property
    def boundaries(self) -> list[tuple[float, float]]:
        """Brackets ``(theta_left, theta_right)`` between consecutive grid
        points whose selection identity differs — the selection-regime
        boundaries resolved at the grid's spacing."""
        thetas = self.df.index.to_numpy()
        regimes = self.df["regime"].to_numpy()
        return [
            (float(thetas[i]), float(thetas[i + 1]))
            for i in range(len(thetas) - 1)
            if regimes[i + 1] != regimes[i]
        ]


def landscape_section(
    builder: Callable[[float], Any],
    state: tuple[int, ...],
    grid: Iterable[float],
    *,
    subset: Sequence[int] | None = None,
    formalism: str | None = None,
    progress: bool | None = None,
) -> LandscapeSection:
    """Evaluate the system irreducibility analysis along a parameter grid.

    Parameters
    ----------
    builder : Callable[[float], Substrate]
        The parameter axis: maps a parameter value θ to a substrate.
        See :func:`weight_axis` for the single-weight case.
    state : tuple[int, ...]
        The substrate state to analyze at every point.
    grid : Iterable[float]
        The parameter values to evaluate, in the order given.
    subset : Sequence[int], optional
        Node indices of the candidate system; ``None`` uses the whole
        substrate.
    formalism : str, optional
        A formalism preset name applied for these evaluations only, as in
        :func:`pyphi.analyze.analyze`.
    progress : bool, optional
        Show a progress bar; defaults to
        ``config.infrastructure.progress_bars``.

    Returns
    -------
    LandscapeSection

    Notes
    -----
    Grid points whose state is dynamically unreachable are skipped and
    recorded on :attr:`LandscapeSection.skipped` rather than raised — a
    parameter grid is an enumeration, and reachability can change along
    it. A reducible point (null analysis) is a row, not a skip:
    reducibility is a real landscape value.

    Points are evaluated sequentially; each point is one full SIA.
    """
    # ponytail: sequential-only; if n >= 5 landscapes demand it, follow
    # pyphi/sweep.py's _run_cells_parallel pattern (module-level cell
    # function, parallel=False inside workers).
    thetas = [float(theta) for theta in grid]
    show_progress = fallback(progress, config.infrastructure.progress_bars)
    iterator = tqdm(
        thetas, desc="Evaluating landscape section", disable=not show_progress
    )

    kept_thetas: list[float] = []
    rows: list[dict[str, Any]] = []
    sias: list[Any] = []
    skipped: list[float] = []
    for theta in iterator:
        try:
            row, sia = _eval_point(builder, theta, state, subset, formalism)
        except _UNREACHABLE:
            skipped.append(theta)
            continue
        kept_thetas.append(theta)
        rows.append(row)
        sias.append(sia)

    regime = 0
    previous: tuple[Any, ...] | None = None
    for row in rows:
        identity = _selection_identity(row)
        if previous is not None and identity != previous:
            regime += 1
        row["regime"] = regime
        previous = identity

    df = pd.DataFrame(rows, index=pd.Index(kept_thetas, name="theta"))
    return LandscapeSection(df=df, sias=sias, skipped=skipped)


@dataclass(frozen=True)
class Perturbation:
    """Local finite-difference behavior of one quantity at one point.

    Built from exactly three analyses, at θ − h, θ, and θ + h.

    Attributes
    ----------
    theta : float
        The evaluation point.
    h : float
        The finite-difference step.
    quantity : str
        The differentiated quantity (one of ``phi``, ``signed_phi``,
        ``normalized_phi``, ``signed_normalized_phi``).
    value : float
        The quantity at θ.
    derivative : float
        Central difference ``(f(θ+h) − f(θ−h)) / 2h``.
    left_derivative, right_derivative : float
        One-sided differences. When they disagree materially, the point is
        near a kink or a selection switch and the central estimate is not
        trustworthy — check :attr:`same_regime`.
    margins : dict[str, float]
        The selection margins at θ, keyed ``"partition"``,
        ``"cause_state"``, ``"effect_state"``; ``NaN`` where undefined.
    margin_derivatives : dict[str, float]
        Central differences of the margins (``NaN`` where a margin is
        undefined at any of the three points).
    same_regime : bool
        Whether all three points share one selection identity (same MIP
        partition and specified states). When ``False``, the central
        derivative straddles a regime boundary; trust only the one-sided
        values, and only on their own sides.
    sias : tuple
        The raw analyses at (θ − h, θ, θ + h).
    """

    theta: float
    h: float
    quantity: str
    value: float
    derivative: float
    left_derivative: float
    right_derivative: float
    margins: dict[str, float]
    margin_derivatives: dict[str, float]
    same_regime: bool
    sias: tuple[Any, Any, Any]

    @property
    def switch_distances(self) -> dict[str, float]:
        """Linearized distance, in parameter units, to each kind of
        selection switch: ``margin / |d(margin)/dθ|``.

        A first-order estimate of how far the parameter can move before
        the corresponding selection changes its winner — the distance at
        which the margin would reach zero if it kept shrinking at its
        current rate. ``NaN`` where the margin or its derivative is
        undefined; ``inf`` where the margin is locally flat.
        """
        distances: dict[str, float] = {}
        for name in _MARGIN_NAMES:
            margin = self.margins[name]
            slope = self.margin_derivatives[name]
            if math.isnan(margin) or math.isnan(slope):
                distances[name] = math.nan
            elif slope == 0.0:
                distances[name] = 0.0 if margin == 0.0 else math.inf
            else:
                distances[name] = abs(margin / slope)
        return distances


def perturb(
    builder: Callable[[float], Any],
    state: tuple[int, ...],
    theta: float,
    *,
    h: float = 1e-4,
    quantity: str = "signed_phi",
    subset: Sequence[int] | None = None,
    formalism: str | None = None,
) -> Perturbation:
    """Estimate the local derivative of an IIT quantity at one point.

    Parameters
    ----------
    builder : Callable[[float], Substrate]
        The parameter axis (see :func:`weight_axis`).
    state : tuple[int, ...]
        The substrate state to analyze.
    theta : float
        The evaluation point.
    h : float
        Finite-difference step. Derivative estimates are stable across
        several orders of magnitude of ``h`` inside a selection regime;
        the default sits in the middle of that range.
    quantity : str
        Which quantity to differentiate: ``"phi"``, ``"signed_phi"``
        (default), ``"normalized_phi"``, or ``"signed_normalized_phi"``.
        The signed variants carry gradient information where the
        positive-part clamp makes the canonical values exactly flat.
    subset : Sequence[int], optional
        Node indices of the candidate system.
    formalism : str, optional
        A formalism preset name applied for these evaluations only.

    Returns
    -------
    Perturbation

    Raises
    ------
    ValueError
        If ``quantity`` is not one of the four φ variants.
    pyphi.exceptions.StateUnreachableError
        If the state is unreachable at any of the three evaluation
        points — a point query fails loud rather than skipping.
    """
    if quantity not in _QUANTITIES:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of: "
            f"{', '.join(sorted(_QUANTITIES))}"
        )
    points = (theta - h, theta, theta + h)
    rows_and_sias = [
        _eval_point(builder, point, state, subset, formalism) for point in points
    ]
    rows = [row for row, _ in rows_and_sias]
    sias = tuple(sia for _, sia in rows_and_sias)
    left, center, right = (row[quantity] for row in rows)

    margin_keys = {name: f"{name}_margin" for name in _MARGIN_NAMES}
    margins = {name: rows[1][margin_keys[name]] for name in _MARGIN_NAMES}
    margin_derivatives = {
        name: (rows[2][margin_keys[name]] - rows[0][margin_keys[name]]) / (2 * h)
        for name in _MARGIN_NAMES
    }
    identities = {_selection_identity(row) for row in rows}

    return Perturbation(
        theta=float(theta),
        h=float(h),
        quantity=quantity,
        value=center,
        derivative=(right - left) / (2 * h),
        left_derivative=(center - left) / h,
        right_derivative=(right - center) / h,
        margins=margins,
        margin_derivatives=margin_derivatives,
        same_regime=len(identities) == 1,
        sias=sias,  # type: ignore[arg-type]
    )
