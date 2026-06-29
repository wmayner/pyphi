"""Cartesian batch driver: run an IIT computation across states, subsystems,
and formalisms, and collect the results into one tidy DataFrame.

``sweep`` takes a substrate and up to three axes (states, candidate subsets,
formalisms), runs the chosen computation on the cartesian product, and returns
a :class:`SweepResult` holding a long-format DataFrame and the aligned raw
result objects. Each result carries its own configuration snapshot, so a row
is independently reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from pyphi import exceptions
from pyphi import utils
from pyphi.conf import config
from pyphi.conf import presets
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

    cell: tuple[Any, Any]


@dataclass(frozen=True)
class SweepResult:
    """A sweep's tidy table plus the raw results behind it.

    ``df`` has one row per computed cell, indexed by the axes that vary.
    ``results`` holds the raw result objects aligned 1:1 with ``df`` rows.
    ``skipped`` lists the ``(formalism, subset, state)`` cells dropped because
    their state is dynamically unreachable (only when an axis is enumerated via
    ``"all"``; explicit cells fail loud instead).
    """

    df: pd.DataFrame
    results: list[Any]
    skipped: list[tuple[str, tuple, tuple]]

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


def _normalize_formalisms(formalisms: Any) -> list[str]:
    if formalisms is None:
        return [config.formalism.iit.version]
    return list(formalisms)


# ---- per-cell compute + row extraction ----


def _dispatch_compute(system: System, compute: Any) -> Any:
    if compute == "sia":
        return system.sia()
    if compute == "ces":
        return system.ces()
    return compute(system)


def _run_cell(cell: tuple[Any, Any], *, substrate: Any, compute: Any, skip: bool) -> Any:
    """Build the system for one (subset, state) cell and run its computation.

    Module-level and config-free so it is picklable for the process backend;
    the active formalism is installed in the worker via the propagated config
    snapshot, not set here. When ``skip`` is true, an unreachable (uncomputable)
    state yields a :class:`_Skipped` sentinel instead of raising.
    """
    subset, state = cell
    try:
        system = System(substrate, state, node_indices=subset)
        return _dispatch_compute(system, compute)
    except _UNREACHABLE:
        if skip:
            return _Skipped(cell)
        raise


def _row_sia(result: Any) -> dict[str, Any]:
    return {
        "phi": float(result.phi),
        "normalized_phi": float(getattr(result, "normalized_phi", float("nan"))),
        "is_irreducible": utils.is_positive(result.phi),
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


def _run_cells_sequential(
    substrate: Any, formalism: str, cells: list[Any], compute: Any, skip: bool
) -> list[Any]:
    results: list[Any] = []
    with config.override(**presets.by_name[formalism]):
        results = [
            _run_cell(c, substrate=substrate, compute=compute, skip=skip) for c in cells
        ]
    return results


def _run_cells_parallel(
    substrate: Any,
    formalism: str,
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
    cell_fn = partial(_run_cell, substrate=substrate, compute=compute, skip=skip)
    results: list[Any] = []
    # Install the formalism and disable inner parallelism in the worker config
    # snapshot the process backend captures; the outer map_reduce parallelizes
    # via its explicit parallel=True (one level of parallelism, no oversubscription).
    with config.override(**presets.by_name[formalism], parallel=False):
        results = map_reduce(
            cell_fn,
            cells,
            parallel=True,
            ordered=True,
            reduce_func=list,
            progress=show,
            desc=f"sweep[{formalism}]",
        )
    if len(results) != len(cells):
        raise AssertionError(
            "map_reduce reducer flattened cell results; expected one per cell"
        )
    return results


def _build_df(
    keys: list[tuple[str, tuple, tuple]],
    rows: list[dict[str, Any]],
    formalisms: list[str],
    subsets: list[tuple],
    states: list[tuple],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    levels: dict[str, list[Any]] = {}
    if len(formalisms) > 1:
        levels["formalism"] = [k[0] for k in keys]
    else:
        df["formalism"] = formalisms[0]
    if len(subsets) > 1:
        levels["subset"] = [k[1] for k in keys]
    else:
        df["subset"] = [subsets[0]] * len(df)
    if len(states) > 1:
        levels["state"] = [k[2] for k in keys]
    else:
        df["state"] = [states[0]] * len(df)
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
    substrate: Any,
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

    Args:
        substrate: the substrate to sweep over.
        states: a state tuple, an iterable of states, or ``"all"``.
        subsets: ``"full"`` (whole system), ``"all"`` (non-empty powerset), or
            an iterable of node-index tuples.
        formalisms: ``None`` (the active formalism) or an iterable of version
            names (``"IIT_3_0"``, ``"IIT_4_0_2023"``, ``"IIT_4_0_2026"``).
        compute: ``"sia"`` (default), ``"ces"``, or a callable taking a
            ``System``.
        parallel: ``None`` follows ``config.infrastructure.parallel``; ``True``/
            ``False`` forces.
        progress: ``None`` follows config; ``True``/``False`` forces.
        seed: stamped into each result's provenance (a bookkeeping label).
    """
    # Auto-enumerated axes ("all") may produce dynamically-unreachable
    # (uncomputable) cells; skip and record those. When every axis is given
    # explicitly, an uncomputable cell fails loud.
    skip_uncomputable = states == "all" or subsets == "all"
    states_ = _normalize_states(substrate, states)
    subsets_ = _normalize_subsets(substrate, subsets)
    formalisms_ = _normalize_formalisms(formalisms)
    use_parallel = config.infrastructure.parallel if parallel is None else parallel

    keys: list[tuple[str, tuple, tuple]] = []
    raw: list[Any] = []
    skipped: list[tuple[str, tuple, tuple]] = []
    for formalism in formalisms_:
        cells = list(product(subsets_, states_))
        if use_parallel:
            results = _run_cells_parallel(
                substrate, formalism, cells, compute, skip_uncomputable, progress
            )
        else:
            results = _run_cells_sequential(
                substrate, formalism, cells, compute, skip_uncomputable
            )
        for (subset, state), result in zip(cells, results, strict=True):
            if isinstance(result, _Skipped):
                skipped.append((formalism, subset, state))
            else:
                keys.append((formalism, subset, state))
                raw.append(result)

    if seed is not None:
        for result in raw:
            with_provenance = getattr(result, "with_provenance", None)
            if with_provenance is not None:
                with_provenance(seed=seed)

    rows = [_extract_row(result, compute) for result in raw]
    df = _build_df(keys, rows, formalisms_, subsets_, states_)
    return SweepResult(df=df, results=raw, skipped=skipped)
