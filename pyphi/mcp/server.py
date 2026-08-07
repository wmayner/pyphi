"""The PyPhi Model Context Protocol server.

Defines the FastMCP application, the tools that wrap PyPhi's public API, and the
``main`` entry point for the ``pyphi-mcp`` console script. The teaching
resources and prompts are registered from :mod:`pyphi.mcp.resources` and
:mod:`pyphi.mcp.prompts`.

The server holds built substrates and analysis results in an in-process
registry keyed by short handles, so a substrate is built once and then explored
across many states and formalism versions without resending its transition
probability matrix.
"""

from __future__ import annotations

import itertools
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

import pyphi
from pyphi import examples
from pyphi import serialize
from pyphi.conf import presets
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.cost import estimate_analysis
from pyphi.cost import runtime_seconds
from pyphi.models.distinctions import Distinctions

from . import content

# Soft guards against accidentally starting an hours-long run, not hard limits
# on what PyPhi can compute. The guard counts the requested analysis's workload
# with pyphi.cost.estimate_analysis, so the partition schemes, connectivity,
# and alphabet all inform the refusal. Each limit governs one work axis, and
# every axis the requested analysis walks is checked: a cause-effect structure
# under IIT 4.0 computes a system irreducibility analysis before unfolding
# distinctions, so it answers to both.
#
# _SIA_PARTITION_LIMIT is the DIRECTED_SET_PARTITION count for 9 fully
# connected binary units — the largest system-level analysis admitted without
# confirmation. _CES_SWEEP_LIMIT admits a fully connected 6-unit binary
# cause-effect structure under JOINT_PARTITION_ALL (31,938,830 sweeps) and
# refuses the 7-unit one (1,450,456,298 sweeps).
#
# _GUARD_COUNT_BUDGET bounds the guard's own counting work (purview
# evaluations plus fresh partition enumerations; memoized counts are free),
# keeping the pre-flight to a few seconds. A walk that exceeds it is
# refused conservatively: a workload too large to count cheaply is treated
# as too large to run unconfirmed.
_SIA_PARTITION_LIMIT = 4_419_572
_CES_SWEEP_LIMIT = 100_000_000
_GUARD_COUNT_BUDGET = 3_000_000

# Friendly level names mapped to the per-level parallelization options. The
# recommended default set covers the levels that pay off for a single-system
# analysis; relations and mechanism partitions are excluded because
# parallelizing them was measured not to pay (see the "parallelization"
# reference topic).
_PARALLEL_LEVELS = {
    "partitions": "parallel_partition_evaluation",
    "purviews": "parallel_purview_evaluation",
    "distinctions": "parallel_distinction_evaluation",
    "complexes": "parallel_complex_evaluation",
    "mechanism_partitions": "parallel_mechanism_partition_evaluation",
    "relations": "parallel_relation_evaluation",
    "macro_systems": "parallel_macro_system_evaluation",
}
_DEFAULT_PARALLEL_LEVELS = ("partitions", "purviews", "distinctions")


def _parallel_overrides(
    parallel: bool | list[str] | None, workers: int | None = None
) -> dict[str, Any]:
    """Build ``pyphi.config`` option values from tool-level parallel arguments.

    Declarative: an explicit level selection determines the full enabled set,
    switching the named levels on and every other level off. ``None`` leaves
    the current configuration untouched (except ``parallel_workers`` when
    ``workers`` is given); ``False`` closes the global gate, which forces
    every level sequential.
    """
    overrides: dict[str, Any] = {}
    if workers is not None:
        overrides["parallel_workers"] = workers
    if parallel is None:
        return overrides
    if parallel is False:
        overrides["parallel"] = False
        return overrides
    enabled = _DEFAULT_PARALLEL_LEVELS if parallel is True else parallel
    unknown = [name for name in enabled if name not in _PARALLEL_LEVELS]
    if unknown:
        known = ", ".join(_PARALLEL_LEVELS)
        raise ValueError(
            f"Unknown parallel level(s) {', '.join(map(repr, unknown))}. "
            f"Available levels: {known}."
        )
    overrides["parallel"] = True
    for name, option in _PARALLEL_LEVELS.items():
        overrides[option] = {
            **dict(getattr(pyphi.config.infrastructure, option)),
            "parallel": name in enabled,
        }
    return overrides


def _parallel_state() -> dict[str, Any]:
    """Summarize the current parallelization configuration."""
    infra = pyphi.config.infrastructure
    return {
        "parallel": infra.parallel,
        "workers": infra.parallel_workers,
        "backend": infra.parallel_backend,
        "levels": {
            name: {
                "parallel": dict(getattr(infra, option))["parallel"],
                "sequential_threshold": dict(getattr(infra, option))[
                    "sequential_threshold"
                ],
            }
            for name, option in _PARALLEL_LEVELS.items()
        },
        "note": (
            "A level runs in parallel only when the global 'parallel' gate "
            "is on AND the level's own flag is on AND the workload meets "
            "its sequential_threshold. See "
            "get_iit_reference('parallelization')."
        ),
    }


# The gotchas are loaded rather than linked. They are the list of mistakes that
# produce wrong results — reporting φₛ as Φ, reading a state as big-endian — and
# an instruction to go read them is skipped exactly when the task feels small
# enough not to need theory. The remaining topics stay behind
# ``get_iit_reference``; only this one has to be present before the first tool
# call. It costs roughly 1,900 tokens.
_INSTRUCTIONS = f"{content.load('primer')}\n\n{content.load('gotchas')}"

mcp = FastMCP("pyphi", instructions=_INSTRUCTIONS)

_substrates: dict[str, Any] = {}
_results: dict[str, Any] = {}
_substrate_counter = itertools.count(1)
_result_counter = itertools.count(1)


def _register_substrate(substrate: Any) -> str:
    handle = f"sub{next(_substrate_counter)}"
    _substrates[handle] = substrate
    return handle


def _register_result(result: Any) -> str:
    ref = f"res{next(_result_counter)}"
    _results[ref] = result
    return ref


def _get_substrate(handle: str) -> Any:
    try:
        return _substrates[handle]
    except KeyError:
        known = ", ".join(_substrates) or "none"
        raise KeyError(
            f"Unknown substrate handle {handle!r}. Build or load one first. "
            f"Known handles: {known}."
        ) from None


def _substrate_summary(substrate: Any) -> dict[str, Any]:
    return {
        "num_nodes": substrate.size,
        "node_labels": list(map(str, substrate.node_labels)),
        "num_states": int(np.prod(substrate.tpm.shape[:-1]))
        if hasattr(substrate, "tpm")
        else None,
        "connectivity_matrix": np.asarray(substrate.cm).astype(int).tolist(),
    }


def _result_summary(result: Any, formalism: str | None = None) -> dict[str, Any]:
    """Build a compact, JSON-safe summary of an analysis result.

    Reads only the scalar fields (φ, φₛ, counts) so the summary stays small
    even when the underlying result serializes to megabytes. Tolerant of the
    different result types ``analyze`` can return (a full analysis, a system
    irreducibility analysis, a cause-effect structure, or bare distinctions)
    across all three formalism versions.

    ``formalism`` names the version that produced ``result``, for results
    that carry no configuration snapshot of their own; a full analysis and a
    system irreducibility analysis both report their own.

    Notes
    -----
    ``system_phi`` and ``big_phi`` are different quantities and must not be
    reported as one another. ``system_phi`` is φₛ, the system integrated
    information, which decides whether the system exists as one whole (it is
    IIT 3.0's Φ under that formalism). ``big_phi`` is IIT 4.0's Φ, the
    structure integrated information: the sum of φ over the Φ-structure's
    distinctions and relations. The ``formalism`` key says which version
    produced them.

    A bare distinctions result whose congruence is unresolved reports its
    count and φ sum under ``_upper_bound`` key names, because congruence
    filtering only ever removes distinctions — sometimes all of them.
    """
    summary: dict[str, Any] = {"type": type(result).__name__}

    if isinstance(result, Distinctions):
        if formalism is not None:
            summary["formalism"] = formalism
        summary.update(_distinctions_summary(result, formalism))
        return summary

    def add_float(key: str, obj: Any, attr: str) -> None:
        value = getattr(obj, attr, None)
        if value is not None:
            summary[key] = float(value)

    sia = getattr(result, "sia", result)
    version = getattr(
        getattr(getattr(sia, "config", None), "formalism", None), "iit", None
    )
    if version is not None:
        summary["formalism"] = version.version
    add_float("system_phi", sia, "phi")
    add_float("cause_phi", getattr(sia, "cause", None), "phi")
    add_float("effect_phi", getattr(sia, "effect", None), "phi")
    if getattr(sia, "partition", None) is not None:
        from pyphi.models.partitions import concise_partition

        summary["mip"] = concise_partition(sia.partition)

    # A full analysis carries its cause-effect structure on ``.ces``; a bare
    # cause-effect structure result is one itself (it has ``big_phi``).
    ces = getattr(result, "ces", None)
    if ces is None and hasattr(result, "big_phi"):
        ces = result
    if ces is not None:
        # IIT 3.0 has no relations, so its structure has no Φ; the sum of
        # distinction φ is not that formalism's big phi.
        if summary.get("formalism") != "IIT_3_0":
            add_float("big_phi", ces, "big_phi")
        # Under IIT 3.0 the cause-effect structure *is* the distinctions.
        if isinstance(ces, Distinctions):
            summary.update(
                _distinctions_summary(ces, summary.get("formalism", formalism))
            )
            return summary
        add_float("sum_phi_distinctions", ces, "sum_phi_distinctions")
        add_float("sum_phi_relations", ces, "sum_phi_relations")
        if hasattr(ces, "distinctions"):
            summary["num_distinctions"] = len(ces.distinctions)
        if hasattr(ces, "relations"):
            relations = ces.relations
            summary["num_relations"] = (
                relations.num_relations()
                if hasattr(relations, "num_relations")
                else len(relations)
            )
    return summary


def _distinctions_summary(
    distinctions: Any, formalism: str | None = None
) -> dict[str, Any]:
    """Summarize a bare distinctions result, flagging unresolved congruence.

    An IIT 4.0 cause-effect structure keeps only the distinctions congruent
    with the system's specified state. Distinctions that have not been
    through that filter are reported under ``_upper_bound`` key names, so
    their count and φ sum cannot be read as the structure's — filtering can
    remove any number of them, including all. IIT 3.0 has no congruence
    filter, so its distinctions are always the structure's.
    """
    from pyphi.models.distinctions import UnresolvedDistinctions

    count = len(distinctions)
    total = float(distinctions.sum_phi())
    if formalism == "IIT_3_0":
        return {"num_distinctions": count, "sum_phi_distinctions": total}
    if isinstance(distinctions, UnresolvedDistinctions):
        return {
            "congruence": "unresolved",
            "num_distinctions_upper_bound": count,
            "sum_phi_distinctions_upper_bound": total,
            "note": (
                "The system's specified state is tied, and the tie is broken "
                "by the φₛ cascade over the tied cause/effect pairs, which "
                "needs the system-partition search. These distinctions have "
                "not been filtered for congruence, so the count and φ sum are "
                "upper bounds on the cause-effect structure's — filtering can "
                "remove any number of them, including all. Run "
                "compute='ces' for the congruent set."
            ),
        }
    return {
        "congruence": "resolved",
        "num_distinctions": count,
        "sum_phi_distinctions": total,
    }


@mcp.tool()
def list_examples() -> dict[str, str]:
    """List the built-in example substrates that ``load_example`` can load.

    Returns a mapping from each example's name to a one-line description. These
    are the standard networks from the IIT literature (XOR, the basic 3-node
    logic-gate system, the IIT 4.0 paper figures, and more).
    """
    out = {}
    for name, func in sorted(examples.EXAMPLES["substrate"].items()):
        doc = (func.__doc__ or "").strip().splitlines()
        out[name] = doc[0].strip() if doc else "(no description)"
    return out


@mcp.tool()
def load_example(name: str) -> dict[str, Any]:
    """Load a built-in example substrate and return a handle for it.

    Parameters
    ----------
    name : str
        An example name from ``list_examples`` (e.g. ``"basic"``, ``"xor"``).

    Returns
    -------
    dict
        The substrate ``handle`` (pass it to ``analyze``/``describe_substrate``)
        and a summary of its nodes and connectivity.
    """
    try:
        func = examples.EXAMPLES["substrate"][name]
    except KeyError:
        known = ", ".join(sorted(examples.EXAMPLES["substrate"]))
        raise KeyError(f"Unknown example {name!r}. Available: {known}.") from None
    substrate = func()
    handle = _register_substrate(substrate)
    return {"handle": handle, **_substrate_summary(substrate)}


@mcp.tool()
def build_substrate(
    tpm: list,
    cm: list | None = None,
    node_labels: list[str] | None = None,
    alphabet: list[int] | None = None,
) -> dict[str, Any]:
    """Build a substrate from a transition probability matrix and return a handle.

    Parameters
    ----------
    tpm : list
        The transition probability matrix, as nested lists. State-by-node form
        (one row per system state, one column per node, giving each node's
        probability of turning on) is the usual input. States are ordered
        little-endian: the FIRST node is the least-significant bit, so state
        (0, 0, 1) is row index 4 in a 3-node system, not row 1.
    cm : list, optional
        The connectivity matrix (``cm[i][j] == 1`` means node i is an input to
        node j). If omitted, full connectivity is assumed — always correct, but
        slower. A *wrong* connectivity matrix produces a wrong Φ, so omit it
        when unsure rather than guessing.
    node_labels : list of str, optional
        Labels for the nodes; defaults to A, B, C, ….
    alphabet : list of int, optional
        The number of states per node, for multi-valued (k-ary) units. Defaults
        to binary. Note that more states does not necessarily mean more Φ.

    Returns
    -------
    dict
        The substrate ``handle`` and a summary of its nodes and connectivity.
    """
    kwargs: dict[str, Any] = {}
    if cm is not None:
        kwargs["cm"] = np.asarray(cm)
    if node_labels is not None:
        kwargs["node_labels"] = node_labels
    if alphabet is not None:
        kwargs["state_space"] = tuple(tuple(range(k)) for k in alphabet)
    substrate = pyphi.Substrate(np.asarray(tpm, dtype=float), **kwargs)
    handle = _register_substrate(substrate)
    return {"handle": handle, **_substrate_summary(substrate)}


@mcp.tool()
def describe_substrate(handle: str) -> dict[str, Any]:
    """Describe a substrate previously loaded or built.

    Parameters
    ----------
    handle : str
        A substrate handle from ``load_example`` or ``build_substrate``.

    Returns
    -------
    dict
        The substrate's nodes, labels, connectivity, state count, and a
        reminder of the little-endian state-index convention.
    """
    substrate = _get_substrate(handle)
    return {
        "handle": handle,
        **_substrate_summary(substrate),
        "state_convention": (
            "States are little-endian: the first node is the "
            "least-significant bit. A state is a tuple of node states, e.g. "
            "(1, 1, 0) means node A on, B on, C off."
        ),
    }


def _refuse_if_large(estimate: Any, compute: str, n_nodes: int) -> None:
    """Raise when an analysis's estimated workload exceeds a soft limit.

    Every axis the analysis actually walks is checked against its own limit.
    Under IIT 4.0 a cause-effect structure computes a system irreducibility
    analysis before unfolding distinctions, so it is charged on both the
    system-partition axis and the mechanism-partition axis; over a sparse
    substrate the first can dominate the second by orders of magnitude. An
    axis the estimate left uncounted is not checked.
    """
    axes = (
        ("system partitions", estimate.system_partitions, _SIA_PARTITION_LIMIT),
        (
            "mechanism-partition sweeps",
            estimate.mechanism_partition_sweeps,
            _CES_SWEEP_LIMIT,
        ),
    )
    over = [
        (axis, count, limit)
        for axis, count, limit in axes
        if count is not None and count > limit
    ]
    if over:
        # Report the axis that overshoots its limit by the widest margin.
        axis, count, limit = max(over, key=lambda item: item[1] / item[2])
        qualifier = "at least " if estimate.capped else ""
        estimated = f"{qualifier}{count:,} {axis} (soft limit {limit:,})"
        blames_sia = axis == "system partitions"
    elif estimate.capped:
        estimated = "beyond the guard's own counting budget"
        blames_sia = estimate.system_partitions is None
    else:
        return

    if blames_sia and compute != "sia":
        cheaper = (
            ", or use compute='distinctions' for the distinctions alone, which "
            "skips the system-partition search"
        )
    elif compute in ("full", "ces"):
        cheaper = ", or use compute='sia' for a cheaper system-level result"
    else:
        # Already the cheapest analysis the tool offers.
        cheaper = ""
    raise ValueError(
        f"A '{compute}' analysis of this {n_nodes}-node substrate is "
        f"estimated at {estimated}; it may run for a very long time. Pass "
        f"confirm_large=true to proceed anyway, use the estimate_cost tool "
        f"to inspect the workload{cheaper}."
    )


@mcp.tool()
def analyze(
    handle: str,
    state: list[int],
    formalism: str | None = None,
    compute: str = "full",
    detail: str = "summary",
    confirm_large: bool = False,
    parallel: bool | list[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run an IIT analysis of a substrate in a state.

    Parameters
    ----------
    handle : str
        A substrate handle from ``load_example`` or ``build_substrate``.
    state : list of int
        The current state, one entry per node, in node order (little-endian).
    formalism : str, optional
        ``"IIT_4_0_2026"`` (default), ``"IIT_4_0_2023"``, or ``"IIT_3_0"``. Each
        defines integrated information differently, so the same substrate and
        state give different values under each. IIT 3.0 has no relations; the
        2026 variant drives a fully deterministic system's φₛ to zero.
    compute : str
        ``"full"`` (default: system integrated information φₛ *and* the full
        Φ-structure), ``"sia"`` (φₛ only), ``"ces"`` (the cause-effect
        structure), or ``"distinctions"`` (the distinctions alone).

        ``"distinctions"`` is the only one that skips the system-partition
        search, which under IIT 4.0 a cause-effect structure otherwise runs
        before unfolding anything; over a sparse substrate that search is
        almost the entire cost. It reports ``congruence`` in the summary:
        ``"resolved"`` means the distinctions are exactly the Φ-structure's,
        while ``"unresolved"`` means the system's specified state is tied and
        the congruent subset is undetermined — the counts then come back
        under ``_upper_bound`` names and must not be reported as the
        structure's.
    detail : str
        ``"summary"`` (default: a readable card plus scalar values) or
        ``"full"`` (also embeds the complete serialized result, which can be
        megabytes for a Φ-structure — prefer ``inspect`` to drill in instead).
    confirm_large : bool
        An analysis whose estimated workload exceeds a soft limit is refused
        unless this is set, to avoid accidentally starting an hours-long
        computation. The workload is counted by the same machinery as the
        ``estimate_cost`` tool, so the partition schemes, connectivity, and
        alphabet all inform the guard. Parallelism does not lift the
        threshold — it divides the constants, not the exponents.
    parallel : bool or list of str, optional
        Parallelism for this call only. ``None`` (default) uses the server's
        current configuration (see ``configure_parallel``); ``true`` runs on
        multiple cores at the recommended levels (``"partitions"``,
        ``"purviews"``, ``"distinctions"``); a list of level names picks the
        levels explicitly; ``false`` forces the call fully sequential.
        Parallelism never changes the result, and workloads below a level's
        sequential threshold run sequentially regardless. Read
        ``get_iit_reference("parallelization")`` for which levels pay off.
    workers : int, optional
        Worker-process count for this call. Default uses the server's
        configuration (-1 = all cores).

    Returns
    -------
    dict
        A ``card`` (human-readable text), a ``summary`` of scalar quantities,
        and a ``result_ref`` for ``inspect``. Φ=0 means the system is
        *reducible*, not that it has no structure.
    """
    substrate = _get_substrate(handle)
    compute_arg = None if compute == "full" else compute
    if not confirm_large:
        overrides = presets.by_name.get(formalism, {}) if formalism else {}
        with pyphi.config.override(**overrides):
            estimate = estimate_analysis(
                substrate, compute=compute_arg, limit=_GUARD_COUNT_BUDGET
            )
        _refuse_if_large(estimate, compute, substrate.size)

    with pyphi.config.override(**_parallel_overrides(parallel, workers)):
        result = pyphi.analyze(
            substrate, tuple(state), formalism=formalism, compute=compute_arg
        )
    ref = _register_result(result)

    out: dict[str, Any] = {
        "result_ref": ref,
        "card": str(result),
        "summary": _result_summary(
            result, formalism or pyphi.config.formalism.iit.version
        ),
    }
    if detail == "full":
        target = getattr(result, "ces", result)
        out["serialized"] = serialize.dumps(target).decode("utf-8")
    return out


@mcp.tool()
def estimate_cost(
    handle: str,
    compute: str = "full",
    formalism: str | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count the workload of an analysis before running it.

    Reports what ``analyze`` would evaluate — system partitions, candidate
    mechanisms, connectivity-pruned purview evaluations, and
    mechanism-partition sweeps — without computing any φ. The counts
    reflect the partition schemes, connectivity, and alphabet under the
    requested formalism.

    ``estimated_cpu_seconds`` converts the distinction axis to time at
    ``pyphi.cost.SECONDS_PER_UNIT``, which is calibrated on reference
    hardware and holds to roughly 20% there. It excludes the
    system-partition axis, whose cost per partition is not calibrated
    against the same unit, and it is ``None`` when the distinction axis was
    not counted.

    Parameters
    ----------
    handle : str
        A substrate handle from ``load_example`` or ``build_substrate``.
    compute : str
        ``"full"`` (default), ``"sia"``, ``"ces"``, or ``"distinctions"`` —
        the analysis whose workload to estimate, as in ``analyze``. Under
        IIT 4.0, ``"ces"`` is charged on the system-partition axis too,
        since unfolding a cause-effect structure computes a system
        irreducibility analysis first.
    formalism : str, optional
        As in ``analyze``.

    Returns
    -------
    dict
        A ``card`` (human-readable text), an ``estimate`` mapping with the
        counts, and ``estimated_cpu_seconds``; ``capped=true`` marks counts
        that are lower bounds.
    """
    substrate = _get_substrate(handle)
    if formalism is not None and formalism not in presets.by_name:
        valid = ", ".join(sorted(presets.by_name))
        raise ValueError(f"unknown formalism {formalism!r}; expected one of: {valid}")
    overrides = presets.by_name[formalism] if formalism is not None else {}
    with pyphi.config.override(**overrides):
        estimate = estimate_analysis(
            substrate,
            compute=None if compute == "full" else compute,
            scope=_scope_from_json(scope, substrate),
        )
    units = estimate.distinction_units
    return {
        "card": str(estimate),
        "estimate": asdict(estimate),
        "estimated_cpu_seconds": (None if units is None else runtime_seconds(units)),
    }


def _scope_from_json(scope: dict[str, Any] | None, substrate: Any) -> Any:
    """Build a resolved CESScope from the tools' JSON scope shape.

    The shape mirrors the scope objects:
    ``{"mechanisms": {"max_order": 2, "containing": [0]},
    "cause_purviews": {...}, "effect_purviews": {...}}`` — each axis with
    any of ``explicit`` (list of unit lists), ``min_order``, ``max_order``,
    ``containing``, ``within``. Units may be labels or indices.
    """
    if scope is None:
        return None
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope
    from pyphi.campaign.scope import resolve_scope

    def axis(d: dict[str, Any] | None) -> AxisScope:
        if not d:
            return AxisScope()
        return AxisScope(
            explicit=None
            if d.get("explicit") is None
            else tuple(tuple(e) for e in d["explicit"]),
            min_order=d.get("min_order"),
            max_order=d.get("max_order"),
            containing=None if d.get("containing") is None else tuple(d["containing"]),
            within=None if d.get("within") is None else tuple(d["within"]),
        )

    built = CESScope(
        mechanisms=axis(scope.get("mechanisms")),
        cause_purviews=axis(scope.get("cause_purviews")),
        effect_purviews=axis(scope.get("effect_purviews")),
    )
    return resolve_scope(built, substrate.node_labels)


@mcp.tool()
def prepare_ces_campaign(
    handle: str,
    state: list[int],
    directory: str,
    units_per_job: float,
    subset: list[int] | None = None,
    scope: dict[str, Any] | None = None,
    formalism: str | None = None,
    sia_ref: str | None = None,
    ordering: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Materialize one system's scoped CES analysis as an HTCondor campaign.

    Plans shards for the scoped distinction computation (and the system
    irreducibility analysis, unless ``sia_ref`` supplies a precomputed
    one), descending mechanism → purview-range → partition-stride to meet
    the per-job budget. Each shard requests memory sized to its largest
    purview repertoire. Submit the directory with ``condor_submit
    pyphi.sub``; monitor with ``campaign_status``; reassemble with
    ``collect_campaign``. See the ``campaigns`` reference topic. Sweeps
    over many states or substrates under one scope are a library-level
    feature of ``pyphi.campaign.prepare_ces``.

    Parameters
    ----------
    handle : str
        A substrate handle from ``load_example`` or ``build_substrate``.
    state : list of int
        The system state.
    directory : str
        Target campaign directory; must not already exist.
    units_per_job : float
        Target work units per shard.
    subset : list of int, optional
        Candidate-system node indices; the whole substrate when omitted.
    scope : dict, optional
        The feasibility surface: per-axis constraint objects as documented
        on ``estimate_cost``.
    formalism : str, optional
        As in ``analyze``.
    sia_ref : str, optional
        A result handle holding a precomputed system irreducibility
        analysis; suppresses SIA shards.
    ordering : str, optional
        ``"bottleneck_first"`` to evaluate likely-reducible partitions
        first within each stride (sparse substrates short-circuit sooner).
    limit : int, optional
        Work budget for the planning walk; raise it for large scoped
        systems whose walk exceeds the default.
    seed : int, optional
        Recorded in the manifest and stamped into provenance at collection.

    Returns
    -------
    dict
        A ``card`` and a ``status`` mapping with the task ledger.
    """
    from pyphi import campaign

    substrate = _get_substrate(handle)
    kwargs: dict[str, Any] = {}
    if limit is not None:
        kwargs["limit"] = limit
    result = campaign.prepare_ces(
        substrate,
        states=tuple(state),
        subsets="full" if subset is None else [tuple(subset)],
        scope=_scope_from_json(scope, substrate),
        directory=directory,
        units_per_job=units_per_job,
        formalisms=formalism,
        sia=None if sia_ref is None else _get_result(sia_ref),
        ordering=ordering,
        seed=seed,
        **kwargs,
    )
    return {"card": str(result), "status": asdict(result)}


@mcp.tool()
def prepare_campaign(
    handles: list[str],
    directory: str,
    states: Any = "all",
    subsets: Any = "full",
    formalisms: list[str] | None = None,
    compute: str = "sia",
    jobs: int | None = None,
    units_per_job: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Materialize a sweep as an HTCondor campaign directory.

    Enumerates the sweep cells over the given substrate handles, packs them
    into cost-balanced condor tasks, and writes a self-contained campaign
    directory (task files, substrates, submit file). The user submits it
    with ``condor_submit pyphi.sub``; monitor with ``campaign_status`` and
    reassemble results with ``collect_campaign``. See the ``campaigns``
    reference topic for the workflow.

    Parameters
    ----------
    handles : list of str
        Substrate handles from ``load_example`` or ``build_substrate``; the
        handle strings become the substrate labels in the result.
    directory : str
        Target campaign directory; must not already exist.
    states, subsets, formalisms, compute
        Sweep axes: explicit lists, or ``"all"`` / ``"full"``.
    jobs : int, optional
        Pack into exactly this many cost-balanced tasks.
    units_per_job : float, optional
        Target work units per task (mutually exclusive with ``jobs``).
    seed : int, optional
        Recorded in the manifest and stamped into provenance at collection.

    Returns
    -------
    dict
        A ``card`` (human-readable summary) and a ``status`` mapping with
        the task ledger.
    """
    from pyphi import campaign

    substrates = {handle: _get_substrate(handle) for handle in handles}
    states_ = states if isinstance(states, str) else [tuple(s) for s in states]
    subsets_ = subsets if isinstance(subsets, str) else [tuple(s) for s in subsets]
    result = campaign.prepare(
        substrates,
        states=states_,
        subsets=subsets_,
        formalisms=formalisms,
        compute=compute,
        directory=directory,
        jobs=jobs,
        units_per_job=units_per_job,
        seed=seed,
    )
    return {"card": str(result), "status": asdict(result)}


@mcp.tool()
def campaign_status(directory: str) -> dict[str, Any]:
    """Report a campaign's task ledger and refresh its resubmission list.

    Classifies every task purely from the campaign directory's output
    files — done, failed, or pending — and rewrites ``remaining.txt`` so
    that resubmitting is running ``condor_submit pyphi.sub`` again.

    Parameters
    ----------
    directory : str
        A campaign directory written by ``prepare_campaign``.

    Returns
    -------
    dict
        A ``card`` and a ``status`` mapping with the task ledger.
    """
    from pyphi import campaign

    result = campaign.status(directory)
    return {"card": str(result), "status": asdict(result)}


@mcp.tool()
def collect_campaign(directory: str, partial: bool = False) -> dict[str, Any]:
    """Reassemble a campaign's outputs into a sweep result.

    Returns the identical result a local sweep over the same axes would
    produce, registered as a result handle for ``inspect``.

    Parameters
    ----------
    directory : str
        A campaign directory whose tasks have been executed.
    partial : bool
        Return the result built from completed tasks even when some are
        missing or failed (default: raise with a per-task summary).

    Returns
    -------
    dict
        The ``result_ref`` handle, the number of collected ``rows``, and
        the number of ``skipped`` cells.
    """
    import dataclasses

    from pyphi import campaign
    from pyphi.sweep import SweepResult

    result = campaign.collect(directory, partial=partial)
    ref = _register_result(result)
    if isinstance(result, SweepResult):
        return {
            "result_ref": ref,
            "rows": len(result.df),
            "skipped": len(result.skipped),
        }
    return {
        "result_ref": ref,
        "type": type(result).__name__,
        "summary": _result_summary(result),
        "scope_report": dataclasses.asdict(campaign.scope_report(directory)),
    }


@mcp.tool()
def configure_parallel(
    enable: bool | None = None,
    levels: list[str] | None = None,
    workers: int | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """Read or set the server's parallelization configuration.

    Settings persist for the life of the server process; a per-call
    ``analyze(parallel=...)`` argument takes precedence for that call. With no
    arguments, reports the current state without changing anything.

    Parameters
    ----------
    enable : bool, optional
        ``true`` opens the global gate and switches on ``levels`` (or the
        recommended set — ``"partitions"``, ``"purviews"``,
        ``"distinctions"`` — when ``levels`` is omitted), switching every
        other level off. ``false`` closes the global gate, which disables all
        parallelism regardless of the per-level flags.
    levels : list of str, optional
        The levels to switch on: ``"partitions"``, ``"purviews"``,
        ``"distinctions"``, ``"complexes"``, ``"mechanism_partitions"``,
        ``"relations"``, ``"macro_systems"``. Implies ``enable=true``. Read
        ``get_iit_reference("parallelization")`` for which levels pay off for
        which workloads.
    workers : int, optional
        Worker-process count (-1 = all cores). On its own, changes only the
        worker count.
    reset : bool
        Restore every parallelization option to PyPhi's defaults (all
        parallelism off), ignoring the other arguments.

    Returns
    -------
    dict
        The resulting configuration: the global gate, worker count, backend,
        and each level's flag and sequential threshold.
    """
    if reset:
        defaults = InfrastructureConfig()
        pyphi.config.parallel = defaults.parallel
        pyphi.config.parallel_workers = defaults.parallel_workers
        pyphi.config.parallel_backend = defaults.parallel_backend
        for option in _PARALLEL_LEVELS.values():
            setattr(pyphi.config, option, dict(getattr(defaults, option)))
    else:
        spec: bool | list[str] | None
        if enable is False:
            spec = False
        elif levels is not None:
            spec = levels
        else:
            spec = enable
        for option, value in _parallel_overrides(spec, workers).items():
            setattr(pyphi.config, option, value)
    return _parallel_state()


@mcp.tool()
def inspect(result_ref: str, path: str = "") -> dict[str, Any]:
    """Inspect one part of a stored analysis result in full detail.

    Parameters
    ----------
    result_ref : str
        A ``result_ref`` returned by ``analyze``.
    path : str
        Which slice to inspect. Empty returns the top-level summary. Otherwise
        a dotted/indexed path into the result, e.g. ``"sia"``, ``"ces"``,
        ``"ces.distinctions[0]"``, ``"ces.relations"``.

    Returns
    -------
    dict
        The compact repr of the selected object and, when the object supports
        it, its full serialization. Relation aggregates (counts, Σφ_r) come
        from PyPhi's analytical path, which does not enumerate every relation.
    """
    try:
        result = _results[result_ref]
    except KeyError:
        known = ", ".join(_results) or "none"
        raise KeyError(
            f"Unknown result_ref {result_ref!r}. Run analyze first. Known refs: {known}."
        ) from None

    obj = _resolve_path(result, path) if path else result
    out: dict[str, Any] = {"path": path or "(root)", "type": type(obj).__name__}
    compact = getattr(obj, "_compact_repr", None)
    out["repr"] = compact() if callable(compact) else repr(obj)
    try:
        out["serialized"] = serialize.dumps(obj).decode("utf-8")
    except TypeError:
        out["serialized"] = None
        out["note"] = "This object has no full serialization; see 'repr'."
    return out


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted, optionally indexed path into a result object."""
    for part in path.split("."):
        name, _, index = part.partition("[")
        obj = getattr(obj, name)
        if index:
            obj = obj[int(index.rstrip("]"))]
    return obj


# The visualizations PyPhi already provides, and what each one needs.
_PLOT_KINDS = {
    "ces": "the cause-effect structure (Φ-structure) — needs a result_ref",
    "repertoires": "the cause and effect repertoires — needs a result_ref",
    "connectivity": "the substrate's causal connectivity graph — needs a handle",
    "tpm": "the state-by-state transition probability matrix — needs a handle",
}

# The views plot_ces offers; keep in sync with pyphi.visualize.plot_ces.
_CES_VIEWS = ("lattice", "hypergraph", "scatter", "matrix", "spectrum")


def _get_result(result_ref: str) -> Any:
    try:
        return _results[result_ref]
    except KeyError:
        known = ", ".join(_results) or "none"
        raise KeyError(
            f"Unknown result_ref {result_ref!r}. Run analyze first. Known refs: {known}."
        ) from None


def _state_by_state(substrate: Any) -> Any:
    """Return a substrate's 2-D state-by-state transition probability matrix.

    Valid for any per-unit alphabet: the (current, next) entry is the product
    over units of each unit's next-state probability given the current joint
    state, with states enumerated in little-endian mixed-radix order
    (``pyphi.utils.all_states``). This is what ``visualize.plot_tpm`` expects.
    """
    from pyphi import utils

    joint = np.asarray(substrate.tpm.to_joint())
    states = list(utils.all_states(substrate.tpm.alphabet_sizes))
    sbs = np.empty((len(states), len(states)))
    for i, current in enumerate(states):
        per_unit = joint[current]  # (unit, next-state) probabilities
        for j, nxt in enumerate(states):
            sbs[i, j] = np.prod([per_unit[u, s] for u, s in enumerate(nxt)])
    return sbs


@mcp.tool()
def plot(
    target: str,
    kind: str = "ces",
    view: str = "lattice",
    max_relations: int | None = None,
) -> Any:
    """Render one of PyPhi's built-in visualizations.

    Requires the ``visualize`` extra (``pip install pyphi[visualize]``). PyPhi
    already provides these figures, so this exposes them rather than
    reconstructing anything. The interactive ``"ces"`` plot is returned as a
    path to a self-contained HTML file to open in a browser (it cannot be shown
    inline); the static figures are returned as an inline PNG.

    Parameters
    ----------
    target : str
        A ``result_ref`` from ``analyze`` for ``"ces"`` and ``"repertoires"``,
        or a substrate handle for ``"connectivity"`` and ``"tpm"``.
    kind : str
        ``"ces"`` — the cause-effect structure (Φ-structure), interactive.
        ``"repertoires"`` — the cause and effect repertoires of the analysis.
        ``"connectivity"`` — the substrate's causal connectivity graph.
        ``"tpm"`` — the state-by-state transition probability matrix.
    view : str
        For ``kind="ces"`` only, which of the five views to draw: ``"lattice"``
        (default, the inclusion Hasse diagram), ``"hypergraph"`` (the 3-D cause
        and effect purviews with relation faces), ``"scatter"``, ``"matrix"``,
        or ``"spectrum"``. (``"barycentric"`` is a *layout*, not a view.)
    max_relations : int, optional
        For ``kind="ces"`` only, draw just the strongest this-many relations by
        φ_r. When the structure's relations are computed analytically
        (``relation_computation="ANALYTICAL"``), whose relation set cannot be
        enumerated, ``None`` draws the strongest 1000; node sizes and the
        spectrum view stay exact regardless. With enumerable (``"CONCRETE"``)
        relations, ``None`` draws every relation. For the full direct-Python
        surface see ``get_iit_reference("visualization")``.

    Returns
    -------
    For ``"ces"``, a message with the path to the interactive HTML file. For
    the static figures, a message with the PNG path plus an inline preview.
    """
    try:
        from pyphi import visualize
    except Exception as error:  # surface the install hint
        return (
            "Visualization requires the optional dependency. Install it with "
            f"'pip install pyphi[visualize]'. ({error})"
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if kind != "ces" and (view != "lattice" or max_relations is not None):
        raise ValueError("view and max_relations apply only to kind='ces'.")

    if kind == "ces":
        if view not in _CES_VIEWS:
            views = ", ".join(_CES_VIEWS)
            raise ValueError(f"Unknown view {view!r}; use one of: {views}.")
        result = _get_result(target)
        fig = visualize.plot_ces(
            getattr(result, "ces", result), view=view, max_relations=max_relations
        )
    elif kind == "repertoires":
        result = _get_result(target)
        fig = visualize.plot_repertoires(result.system, result.sia)[0]
    elif kind == "connectivity":
        substrate = _get_substrate(target)
        fig = plt.figure()
        visualize.plot_graph(substrate.to_networkx())
    elif kind == "tpm":
        from pyphi import utils

        substrate = _get_substrate(target)
        states = [
            "".join(map(str, s)) for s in utils.all_states(substrate.tpm.alphabet_sizes)
        ]
        fig = visualize.plot_tpm(_state_by_state(substrate), states=states)[0]
    else:
        kinds = ", ".join(_PLOT_KINDS)
        raise ValueError(f"Unknown plot kind {kind!r}; use one of: {kinds}.")

    return _render_figure(fig, kind, plt)


def _render_figure(fig: Any, kind: str, plt: Any) -> Any:
    """Write a plotly or matplotlib figure to disk and return it for the client.

    Interactive (plotly) figures are returned as an HTML path only, with no
    inline image: a static snapshot of a figure meant to be rotated and hovered
    is misleading, and an inline preview would let a reader mistake it for the
    real thing and never open the interactive file. Static (matplotlib) figures
    have no interactive form, so they are returned as an inline PNG.
    """
    from mcp.server.fastmcp import Image

    out_dir = Path(tempfile.gettempdir()) / "pyphi-mcp"
    out_dir.mkdir(exist_ok=True)
    stem = f"{kind}-{uuid.uuid4().hex[:8]}"

    if hasattr(fig, "write_html"):  # plotly: interactive, HTML only
        html_path = out_dir / f"{stem}.html"
        fig.write_html(str(html_path), include_plotlyjs="inline")
        return (
            f"The {kind} plot is interactive — you rotate, zoom, and hover over "
            "its distinctions and relations — so it cannot be shown inline. "
            f"Open this file in a browser to explore it:\n{html_path}"
        )

    # matplotlib: a static PNG, previewed inline
    png_path = out_dir / f"{stem}.png"
    fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
    data = png_path.read_bytes()
    plt.close(fig)
    return [f"Plot written to {png_path}.", Image(data=data, format="png")]


def main(argv: list[str] | None = None) -> int:
    """The ``pyphi-mcp`` entry point.

    With no subcommand, runs the server over stdio. ``install`` and
    ``uninstall`` set the server up in a project instead; see
    :mod:`pyphi.mcp.install`.
    """
    from . import install as install_module

    args = install_module.build_parser().parse_args(argv)
    if args.command is None:
        mcp.run()
        return 0
    return install_module.run(args)


# Assemble the rest of the application. These modules receive ``mcp`` through
# ``register`` rather than importing it, so there is no import cycle.
from . import prompts as _prompts  # noqa: E402
from . import resources as _resources  # noqa: E402

_resources.register(mcp)
_prompts.register(mcp)


if __name__ == "__main__":
    main()
