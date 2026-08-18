"""Cartesian batch driver: run an IIT computation across substrates, states,
subsystems, and formalisms, and collect the results into one tidy DataFrame.

``sweep`` takes one or more substrates and up to three further axes (states,
candidate subsets, formalisms), runs the chosen computation on the cartesian
product, and returns a :class:`SweepResult` holding a long-format DataFrame
and the aligned raw result objects. Each result carries its own configuration
snapshot, so a row is independently reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from pyphi import exceptions
from pyphi import numerics
from pyphi import utils
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.serializable import Serializable
from pyphi.substrate import Substrate
from pyphi.system import System

# A dynamically-unreachable state has no defined cause/effect repertoire, so its
# Φ cannot be computed; these are the errors that signals.
_UNREACHABLE = (
    exceptions.StateUnreachableForwardsError,
    exceptions.StateUnreachableBackwardsError,
)


@dataclass(frozen=True)
class _Skipped:
    """Sentinel returned for a cell whose state is uncomputable (unreachable)."""

    cell: tuple[Any, Any, Any]


@dataclass(frozen=True)
class SweepResult(Serializable):
    """A sweep's tidy table plus the raw results behind it.

    ``df`` has one row per computed cell, indexed by the axes that vary.
    ``results`` holds the raw result objects aligned 1:1 with ``df`` rows.
    ``skipped`` lists the ``(substrate, formalism, subset, state)`` cells
    dropped because their state is dynamically unreachable (only when an axis
    is enumerated via ``"all"``; explicit cells fail loud instead).
    """

    df: pd.DataFrame
    results: list[Any]
    skipped: list[tuple[Any, str, tuple, tuple]]

    def to_pandas(self) -> pd.DataFrame:
        return self.df


# ---- axis normalization ----


def _normalize_states(substrate: Any, states: Any) -> list[tuple[int, ...]]:
    if states == "all":
        return list(utils.all_states(substrate.factored_tpm.alphabet_sizes))
    if isinstance(states, tuple) and all(isinstance(x, int) for x in states):
        return [states]
    return [tuple(s) for s in states]


def _normalize_subsets(substrate: Any, subsets: Any) -> list[tuple[int, ...]]:
    nodes = range(len(substrate))
    if subsets == "full":
        return [tuple(nodes)]
    if subsets == "all":
        return list(utils.powerset(nodes, nonempty=True))
    return [tuple(s) for s in subsets]


def _normalize_formalisms(formalisms: Any) -> list[str | None]:
    """Normalize the formalisms axis; ``None`` means the ambient config.

    ``None`` is kept as a sentinel rather than resolved to the ambient
    version name: an explicit version name applies its complete preset,
    while the ambient config may carry user customizations that a preset
    would silently reset.
    """
    if formalisms is None:
        return [None]
    if isinstance(formalisms, str):
        return [formalisms]
    return list(formalisms)


_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_substrates(substrates: Any) -> list[tuple[Any, Any]]:
    """Normalize the substrates argument to labeled ``(label, substrate)`` pairs.

    A mapping supplies its own labels; a bare substrate gets label ``0``; any
    other iterable is labeled by position.
    """
    if isinstance(substrates, Mapping):
        for label in substrates:
            if not (isinstance(label, str) and _LABEL_RE.match(label)):
                raise ValueError(
                    f"substrate label {label!r} must match [A-Za-z0-9_-]+ "
                    "(labels are used in filenames)"
                )
        return list(substrates.items())
    if isinstance(substrates, Substrate):
        return [(0, substrates)]
    return list(enumerate(substrates))


def _enumerate_cells(
    labeled: list[tuple[Any, Any]],
    states: Any,
    subsets: Any,
    formalisms_: list[str | None],
) -> list[tuple[Any, str | None, tuple, tuple]]:
    """Enumerate ``(label, formalism, subset, state)`` cells in canonical order.

    Explicit ``states``/``subsets`` apply to every substrate; ``"all"`` is
    enumerated per substrate, so substrates of different sizes coexist.
    """
    cells = []
    for formalism in formalisms_:
        for label, substrate in labeled:
            for subset, state in product(
                _normalize_subsets(substrate, subsets),
                _normalize_states(substrate, states),
            ):
                cells.append((label, formalism, subset, state))
    return cells


# ---- per-cell compute + row extraction ----


def _dispatch_compute(system: System, compute: Any) -> Any:
    if compute == "sia":
        return system.sia()
    if compute == "ces":
        return system.ces()
    if callable(compute):
        return compute(system)
    raise ValueError(
        f"unknown compute: {compute!r}; expected 'sia', 'ces', or a callable"
    )


def _run_cell(
    cell: tuple[Any, Any, Any], *, substrates: dict, compute: Any, skip: bool
) -> Any:
    """Build the system for one (label, subset, state) cell and run its computation.

    Module-level and config-free so it is picklable for the process backend;
    the active formalism is installed in the worker via the propagated config
    snapshot, not set here. When ``skip`` is true, an unreachable (uncomputable)
    state yields a :class:`_Skipped` sentinel instead of raising.
    """
    label, subset, state = cell
    try:
        system = System(substrates[label], state, node_indices=subset)
        return _dispatch_compute(system, compute)
    except _UNREACHABLE:
        if skip:
            return _Skipped(cell)
        raise


def _optional_margin(value: Any) -> float | None:
    return None if value is None else float(value)


def _row_sia(result: Any) -> dict[str, Any]:
    # Selection margins exist only on IIT 4.0 SIAs; other formalisms' cells
    # carry None in those columns.
    state_margins = getattr(result, "state_margins", None)
    return {
        "phi": float(result.phi),
        "normalized_phi": float(getattr(result, "normalized_phi", float("nan"))),
        "is_irreducible": numerics.is_positive(result.phi),
        "partition_margin": _optional_margin(getattr(result, "partition_margin", None)),
        "cause_state_margin": _optional_margin(
            state_margins[Direction.CAUSE] if state_margins is not None else None
        ),
        "effect_state_margin": _optional_margin(
            state_margins[Direction.EFFECT] if state_margins is not None else None
        ),
        "effectively_tied": getattr(result, "effectively_tied", None),
    }


def _row_ces(result: Any) -> dict[str, Any]:
    # IIT 4.0: CauseEffectStructure (.sia / .distinctions / .relations).
    # IIT 3.0: Distinctions (tuple-like; no .sia, no relations).
    sia = getattr(result, "sia", None)
    relations = getattr(result, "relations", None)
    distinctions = getattr(result, "distinctions", result)
    return {
        "phi": float(sia.phi) if sia is not None else float("nan"),
        "n_distinctions": len(distinctions),
        "sum_phi_r": (
            float(relations.sum_phi()) if relations is not None else float("nan")
        ),
    }


def _extract_row(result: Any, compute: Any) -> dict[str, Any]:
    if compute == "sia":
        return _row_sia(result)
    if compute == "ces":
        return _row_ces(result)
    to_pandas = getattr(result, "to_pandas", None)
    if to_pandas is not None:
        record = to_pandas()
        if isinstance(record, pd.Series):
            return record.to_dict()
    return {"phi": getattr(result, "phi", None)}


# ---- execution ----


def _formalism_preset(formalism: str | None) -> dict[str, Any]:
    """The config override for a formalism axis value.

    An explicit version name applies its complete preset; ``None`` (the
    ambient config) applies nothing, so user customizations are honored
    exactly as :func:`pyphi.analyze` honors them.
    """
    return {} if formalism is None else dict(presets.by_name[formalism])


def _run_cells_sequential(
    substrates: dict,
    formalism: str | None,
    cells: list[Any],
    compute: Any,
    skip: bool,
    progress: Any = None,
) -> list[Any]:
    resolved_progress = (
        config.infrastructure.progress_bars if progress is None else progress
    )
    results: list[Any] = []
    with config.override(
        **_formalism_preset(formalism), progress_bars=resolved_progress
    ):
        results = [
            _run_cell(c, substrates=substrates, compute=compute, skip=skip)
            for c in cells
        ]
    return results


def _run_cells_parallel(
    substrates: dict,
    formalism: str | None,
    cells: list[Any],
    compute: Any,
    skip: bool,
    progress: Any,
) -> list[Any]:
    from functools import partial

    from pyphi.parallel import map_reduce

    show = config.infrastructure.progress_bars if progress is None else progress
    # partial binds the per-cell args into a picklable callable (a module-level
    # function + picklable args) for the process backend.
    cell_fn = partial(_run_cell, substrates=substrates, compute=compute, skip=skip)
    results: list[Any] = []
    # Install the formalism and disable inner parallelism in the worker config
    # snapshot the process backend captures; the outer map_reduce parallelizes
    # via its explicit parallel=True (one level of parallelism, no oversubscription).
    with config.override(**_formalism_preset(formalism), parallel=False):
        results = map_reduce(
            cell_fn,
            cells,
            parallel=True,
            ordered=True,
            reduce_func=list,
            progress=show,
            desc=f"sweep[{formalism if formalism is not None else 'active config'}]",
            # Cells are whole SIA/CES computations: cost-sampling would run
            # several inline in the parent and discard their results.
            chunksize=1,
        )
    if len(results) != len(cells):
        raise AssertionError(
            "map_reduce reducer flattened cell results; expected one per cell"
        )
    return results


_AXIS_NAMES = ("substrate", "formalism", "subset", "state")


def _build_df(
    keys: list[tuple[Any, str, tuple, tuple]],
    rows: list[dict[str, Any]],
    enumerated: list[tuple[Any, str, tuple, tuple]],
) -> pd.DataFrame:
    """Build the tidy table; an axis is an index level iff it varies.

    ``enumerated`` is the full cell enumeration (before unreachable-state
    skips), so whether an axis varies is a property of what was asked for,
    not of which cells happened to compute.
    """
    df = pd.DataFrame(rows)
    levels: dict[str, list[Any]] = {}
    for pos, name in enumerate(_AXIS_NAMES):
        distinct = {cell[pos] for cell in enumerated}
        if len(distinct) > 1:
            levels[name] = [k[pos] for k in keys]
        else:
            df[name] = [next(iter(distinct))] * len(df)
    if len(levels) == 1:
        name, values = next(iter(levels.items()))
        # tupleize_cols=False keeps tuple state/subset values as scalar index
        # entries instead of expanding them into a MultiIndex.
        df.index = pd.Index(values, name=name, tupleize_cols=False)
    elif len(levels) > 1:
        df.index = pd.MultiIndex.from_arrays(
            list(levels.values()), names=list(levels.keys())
        )
    return df


def sweep(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    compute: Any = "sia",
    parallel: bool | None = None,
    progress: bool | None = None,
    seed: int | None = None,
) -> SweepResult:
    """Run a computation across the cartesian product of axes into a tidy table.

    Parameters
    ----------
    substrates
        A single substrate, a sequence of substrates (labeled by position),
        or a mapping of label to substrate.
    states
        A state tuple, an iterable of states, or ``"all"``. Explicit states
        apply to every substrate; ``"all"`` enumerates per substrate.
    subsets
        ``"full"`` (whole system), ``"all"`` (non-empty powerset), or an
        iterable of node-index tuples. Explicit subsets apply to every
        substrate; ``"full"`` and ``"all"`` are resolved per substrate.
    formalisms
        ``None`` (the active configuration, honored exactly as
        :func:`pyphi.analyze` honors it — no preset is applied, so runtime
        customizations survive) or an iterable of version names
        (``"IIT_3_0"``, ``"IIT_4_0_2023"``, ``"IIT_4_0_2026"``), each of
        which applies its complete preset.
    compute
        ``"sia"`` (default), ``"ces"``, or a callable taking a ``System``.
    parallel : bool or None, optional
        ``None`` follows ``config.infrastructure.parallel``; ``True`` or
        ``False`` forces.
    progress : bool or None, optional
        ``None`` follows config; ``True`` or ``False`` forces.
    seed : int or None, optional
        Stamped into each result's provenance (a bookkeeping label).

    Returns
    -------
    SweepResult
        The tidy long-format table, the aligned raw result objects, and the
        list of cells skipped because their state is dynamically unreachable.

    Notes
    -----
    Cells are skipped (rather than raising) only when an axis is enumerated via
    ``"all"``; when every axis is given explicitly, an uncomputable cell raises.
    """
    # Auto-enumerated axes ("all") may produce dynamically-unreachable
    # (uncomputable) cells; skip and record those. When every axis is given
    # explicitly, an uncomputable cell fails loud.
    skip_uncomputable = states == "all" or subsets == "all"
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    formalisms_ = _normalize_formalisms(formalisms)
    enumerated = _enumerate_cells(labeled, states, subsets, formalisms_)
    use_parallel = config.infrastructure.parallel if parallel is None else parallel

    keys: list[tuple[Any, str, tuple, tuple]] = []
    raw: list[Any] = []
    skipped: list[tuple[Any, str, tuple, tuple]] = []
    for formalism in formalisms_:
        cells = [
            (label, subset, state)
            for label, f, subset, state in enumerated
            if f == formalism
        ]
        if use_parallel:
            results = _run_cells_parallel(
                substrate_map, formalism, cells, compute, skip_uncomputable, progress
            )
        else:
            results = _run_cells_sequential(
                substrate_map, formalism, cells, compute, skip_uncomputable, progress
            )
        # The ambient sentinel executes with no preset applied but is
        # reported under the active version name in the results table.
        reported = config.formalism.iit.version if formalism is None else formalism
        for (label, subset, state), result in zip(cells, results, strict=True):
            if isinstance(result, _Skipped):
                skipped.append((label, reported, subset, state))
            else:
                keys.append((label, reported, subset, state))
                raw.append(result)

    if seed is not None:
        for result in raw:
            with_provenance = getattr(result, "with_provenance", None)
            if with_provenance is not None:
                with_provenance(seed=seed)

    rows = [_extract_row(result, compute) for result in raw]
    labeled_enumeration = [
        (
            label,
            config.formalism.iit.version if f is None else f,
            subset,
            state,
        )
        for label, f, subset, state in enumerated
    ]
    df = _build_df(keys, rows, labeled_enumeration)
    return SweepResult(df=df, results=raw, skipped=skipped)
