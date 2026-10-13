"""Cartesian batch driver: run an IIT computation across substrates, states,
and candidate subsets, and collect the results into one long-format DataFrame.

``sweep`` takes one or more substrates and further axes (states, candidate
subsets, and optionally earlier versions of IIT), runs the chosen computation
on the cartesian product, and returns a :class:`SweepResult` holding a
long-format DataFrame and the aligned raw result objects. Each result carries
its own configuration snapshot, so a row is independently reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from pyphi import exceptions
from pyphi import numerics
from pyphi import utils
from pyphi.analysis import _check_compute
from pyphi.analysis import _run_compute
from pyphi.analysis import _structure_record
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.conf import presets
from pyphi.conf._helpers import check_parallel_flag
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.tables import capped_table
from pyphi.models.ces import CauseEffectStructure
from pyphi.serializable import Serializable
from pyphi.substrate import Substrate
from pyphi.substrate import _coerce_state_to_indices
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


_AXIS_NAMES = ("substrate", "formalism", "subset", "state")


@dataclass(frozen=True, eq=False, repr=False)
class SweepResult(Displayable, Serializable):
    """A sweep's long-format table plus the raw results behind it.

    ``df`` has one row per computed cell and a default ``RangeIndex``. Its
    first four columns are the axes, ``substrate``, ``formalism``,
    ``subset`` and ``state``, whatever the number of values on each; the
    computed quantities follow. ``results`` holds the raw result objects
    aligned 1:1 with ``df`` rows. ``skipped`` lists the ``(substrate,
    formalism, subset, state)`` cells dropped because their state is
    dynamically unreachable (only when an axis is enumerated via ``"all"``;
    explicit cells fail loud instead).
    """

    df: pd.DataFrame
    results: list[Any]
    skipped: list[tuple[Any, str, tuple, tuple]]

    def __eq__(self, other: object) -> bool:
        # The generated dataclass __eq__ would compare ``df`` with ``==``,
        # which yields an elementwise DataFrame whose truth value raises.
        if not isinstance(other, SweepResult):
            return NotImplemented
        return (
            self.df.equals(other.df)
            and self.results == other.results
            and self.skipped == other.skipped
        )

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        df = self.df
        present = [name for name in _AXIS_NAMES if name in df.columns]
        counts = {name: df[name].nunique() for name in present}
        rows = [Row("Rows", len(df)), Row("Skipped", len(self.skipped))]
        if "formalism" in df.columns:
            rows.append(Row("Formalisms", ", ".join(map(str, df.formalism.unique()))))
        rows += [
            Row(f"{name.capitalize()}s", counts[name])
            for name in present
            if name != "formalism"
        ]
        quantities = [c for c in df.columns if c not in _AXIS_NAMES]
        sections = [Section(rows=tuple(rows))]
        if len(df):
            varying = [name for name in present if counts[name] > 1] or present[-1:]
            shown = [
                c for c in ("phi", "big_phi", "n_distinctions", "sum_phi_d") if c in df
            ][:2] or quantities[:2]
            columns = varying + shown
            sections.append(
                Section(
                    label="Table",
                    body=(
                        capped_table(
                            tuple(columns),
                            df[columns].itertuples(index=False),
                            tuple,
                            total=len(df),
                        ),
                    ),
                )
            )
        return Description(
            title="SweepResult",
            sections=tuple(sections),
            compact=f"SweepResult({len(df)} rows, {len(self.skipped)} skipped)",
        )


# ---- axis normalization ----


def _check_mode(axis: str, value: str, modes: tuple[str, ...]) -> None:
    if value not in modes:
        expected = " or ".join(repr(mode) for mode in modes)
        one = "one state tuple" if axis == "states" else "one subset of units"
        raise ValueError(
            f"unknown {axis} {value!r}; expected {expected}, {one}, or an "
            "iterable of them"
        )


def _is_entry(x: Any) -> bool:
    """Whether ``x`` is a whole axis entry (a sequence) rather than one element."""
    return isinstance(x, Iterable) and not isinstance(x, str)


def _explicit_axis(values: Any, axis: str) -> list[tuple[Any, ...]]:
    """An explicit states or subsets axis as a list of tuples.

    A flat sequence, none of whose items is itself a sequence, is one entry,
    as ``analyze(subset=...)`` reads it; anything else is an iterable of
    entries, such as the rows of a 2-D array.
    """
    values = list(values)
    if values and not any(_is_entry(x) for x in values):
        values = [values]
    if not values:
        raise ValueError(f"{axis} is empty")
    for entry in values:
        if not _is_entry(entry):
            hint = (
                "; 'full' and 'all' are valid only as the whole subsets value"
                if entry in ("full", "all")
                else ""
            )
            raise ValueError(
                f"{axis} entry {entry!r} is not a sequence; give one tuple or "
                f"an iterable of tuples{hint}"
            )
    # Numpy scalars become Python scalars, so entries serialize to JSON.
    return [
        tuple(x.item() if isinstance(x, np.generic) else x for x in entry)
        for entry in values
    ]


def _axis_values(axis: Any, name: str) -> Any:
    """Materialize an explicit axis once, so an iterator survives reuse.

    A mode string passes through; ``subsets=None`` means the whole system,
    as ``analyze(subset=None)`` does.
    """
    if axis is None:
        if name == "subsets":
            return "full"
        raise ValueError(f"{name} is None; give a tuple, an iterable of them, or 'all'")
    if isinstance(axis, str):
        return axis
    return _explicit_axis(axis, name)


def _normalize_states(substrate: Any, states: Any) -> list[tuple[int, ...]]:
    if isinstance(states, str):
        _check_mode("states", states, ("all",))
        return list(utils.all_states(substrate.factored_tpm.alphabet_sizes))
    # A state given more than once, however it is spelled, runs once.
    return list(
        dict.fromkeys(
            _coerce_state_to_indices(state, substrate.state_space)
            for state in _explicit_axis(states, "states")
        )
    )


def _normalize_subsets(substrate: Any, subsets: Any) -> list[tuple[int, ...]]:
    nodes = range(len(substrate))
    if isinstance(subsets, str):
        _check_mode("subsets", subsets, ("full", "all"))
        if subsets == "full":
            return [tuple(nodes)]
        return list(utils.powerset(nodes, nonempty=True))
    # Resolved indices, so one system has one key and runs once however it
    # was spelled.
    return list(
        dict.fromkeys(
            substrate.node_labels.coerce_to_indices(subset)
            for subset in _explicit_axis(subsets, "subsets")
        )
    )


def _enumerates(axis: Any) -> bool:
    """Whether an axis argument is enumerated by PyPhi rather than given."""
    return isinstance(axis, str) and axis == "all"


def _normalize_formalisms(formalisms: Any) -> list[str | None]:
    """Normalize the formalisms axis; ``None`` means the ambient config.

    ``None`` means the ambient config; a version name applies its complete
    preset. Duplicates, including two spellings of one version, are dropped.
    """
    if formalisms is None:
        return [None]
    if isinstance(formalisms, (str, Mapping)):
        formalisms = [formalisms]
    # One entry per version, however many spellings name it.
    names = list(dict.fromkeys(presets.canonical(f) for f in formalisms))
    if not names:
        raise ValueError("formalisms is empty; pass None for the active version")
    return names


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
        labeled = list(substrates.items())
    elif isinstance(substrates, Substrate):
        labeled = [(0, substrates)]
    else:
        labeled = list(enumerate(substrates))
    if not labeled:
        raise ValueError("substrates is empty")
    return labeled


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
        system = System(substrates[label], state, subset=subset)
        return _run_compute(system, compute)
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


def _row_structure(result: Any) -> dict[str, Any]:
    # An Analysis carries its cause-effect structure, which embeds the SIA.
    ces = getattr(result, "ces", result)
    return _structure_record(ces.sia, ces)


def _row_distinctions(result: Any) -> dict[str, Any]:
    from pyphi.models.distinctions import UnresolvedDistinctions

    # Unresolved distinctions were not filtered for congruence, so their
    # count and Σφ_d are upper bounds on the structure's.
    return {
        "n_distinctions": len(result),
        "sum_phi_d": float(result.sum_phi()),
        "congruence": (
            "unresolved" if isinstance(result, UnresolvedDistinctions) else "resolved"
        ),
    }


def _extract_row(result: Any, compute: Any) -> dict[str, Any]:
    if compute == "sia":
        return _row_sia(result)
    if compute is None or compute == "ces":
        return _row_structure(result)
    if compute == "distinctions":
        return _row_distinctions(result)
    to_pandas = getattr(result, "to_pandas", None)
    record = to_pandas() if to_pandas is not None else None
    if isinstance(record, pd.DataFrame) and len(record) == 1:
        record = record.iloc[0]
    if isinstance(record, pd.Series):
        return record.to_dict()
    if result is not None and pd.api.types.is_scalar(result):
        return {"value": result}
    return {}


def _long_format(df: pd.DataFrame, results: list[Any]) -> pd.DataFrame:
    """Bring a loaded sweep table to the current layout.

    Files saved by 2.0.0rc1 to rc3 indexed the axes that varied and kept the
    others as trailing columns, and their ``compute="ces"`` rows had only
    ``phi``, ``n_distinctions`` and ``sum_phi_r``; tables in the current
    layout pass through unchanged.
    """
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    if (
        results
        and "big_phi" not in df.columns
        and {"n_distinctions", "sum_phi_r"} <= set(df.columns)
        and all(isinstance(r, CauseEffectStructure) for r in results)
    ):
        rows = pd.DataFrame([_row_structure(r) for r in results])
        df = pd.concat(
            [df.drop(columns=list(rows.columns), errors="ignore"), rows], axis=1
        )
    axes = [name for name in _AXIS_NAMES if name in df.columns]
    return df.reindex(columns=axes + [c for c in df.columns if c not in _AXIS_NAMES])


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


def _build_df(
    keys: list[tuple[Any, str, tuple, tuple]], rows: list[dict[str, Any]]
) -> pd.DataFrame:
    """The long-format table: one column per axis, then the computed
    quantities, with a default ``RangeIndex``."""
    axes = pd.DataFrame(
        {name: [key[i] for key in keys] for i, name in enumerate(_AXIS_NAMES)}
    )
    return pd.concat([axes, pd.DataFrame(rows)], axis=1)


def sweep(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    compute: Any = "sia",
    parallel: bool | None = None,
    progress: bool | None = None,
) -> SweepResult:
    """Run a computation across the cartesian product of axes into a
    long-format table.

    Parameters
    ----------
    substrates
        A single substrate, a sequence of substrates (labeled by position),
        or a mapping of label to substrate.
    states
        A state tuple, an iterable of states (such as the rows of a 2-D
        array or an iterator), or ``"all"``. Explicit states apply to every
        substrate; ``"all"`` enumerates per substrate. The ``state`` column
        holds each state as unit-state indices. A state given more than once
        runs once.
    subsets
        ``"full"`` or ``None`` (whole system), ``"all"`` (non-empty
        powerset), one subset as a tuple of node indices or of labels, or an
        iterable of subsets. Explicit subsets apply to every substrate;
        ``"full"`` and ``"all"`` are resolved per substrate. The ``subset``
        column holds each subset as sorted node indices, however it was
        given, and subsets that name the same units run once.
    formalisms
        Leave unset to compute IIT under the active configuration, honored
        exactly as :func:`pyphi.analyze` honors it (no preset is applied, so
        runtime customizations survive). To reproduce published results,
        pass an iterable of version names (``"IIT_4_0_2023"``, ``"IIT_3_0"``,
        or ``"IIT_4_0_2026"`` for the current version), each of which applies
        its complete preset. A version named more than once, under any of
        its names, runs once. See :doc:`/howto/earlier-versions`.
    compute
        What to compute in each cell, as in :func:`pyphi.analyze`:
        ``"sia"`` (the default), ``"ces"``, ``"distinctions"``, ``None``
        for the full :class:`~pyphi.analysis.Analysis`, or a callable taking
        a ``System``. The default differs from ``analyze``'s because a sweep
        multiplies the cost of one analysis by the number of cells, and the
        system irreducibility analysis is the cheapest. A callable's return
        value is stored in ``results``; the table gets a ``value`` column
        when it is a scalar, or the columns of its ``to_pandas()`` when that
        is one row.
    parallel : bool or None, optional
        ``None`` follows ``config.infrastructure.parallel``; ``True`` or
        ``False`` forces.
    progress : bool or None, optional
        ``None`` follows config; ``True`` or ``False`` forces.

    Returns
    -------
    SweepResult
        The long-format table, the aligned raw result objects, and the
        list of cells skipped because their state is dynamically unreachable.

    Raises
    ------
    ValueError
        Before any computation, if an axis is empty, a string axis is not one
        of its modes, a subset or state is invalid for a substrate, a
        formalism is unknown, or ``compute`` is not one of the values above.

    Notes
    -----
    Cells are skipped (rather than raising) only when an axis is enumerated via
    ``"all"``; when every axis is given explicitly, an uncomputable cell raises.

    The table's columns after the axes depend on ``compute``. ``"sia"``
    rows hold φₛ (``phi``), ``normalized_phi``, ``is_irreducible`` and the
    selection margins. ``"ces"`` and ``None`` rows hold the columns of
    :meth:`Analysis.to_pandas <pyphi.analysis.Analysis.to_pandas>`: φₛ
    (``phi``), ``normalized_phi``, Φ (``big_phi``), ``n_distinctions``,
    ``sum_phi_d`` and ``sum_phi_r``. ``"distinctions"`` rows hold
    ``n_distinctions``, ``sum_phi_d`` and ``congruence``; where
    ``congruence`` is ``"unresolved"`` the system's specified state is tied,
    so the count and the sum are upper bounds on the structure's.
    """
    # Auto-enumerated axes ("all") may produce dynamically-unreachable
    # (uncomputable) cells; skip and record those. When every axis is given
    # explicitly, an uncomputable cell fails loud.
    _check_compute(compute)
    states = _axis_values(states, "states")
    subsets = _axis_values(subsets, "subsets")
    skip_uncomputable = _enumerates(states) or _enumerates(subsets)
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    formalisms_ = _normalize_formalisms(formalisms)
    enumerated = _enumerate_cells(labeled, states, subsets, formalisms_)
    use_parallel = fallback(
        check_parallel_flag(parallel), config.infrastructure.parallel
    )

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

    rows = [_extract_row(result, compute) for result in raw]
    return SweepResult(df=_build_df(keys, rows), results=raw, skipped=skipped)
