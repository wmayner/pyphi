"""Distribute PyPhi computations across an HTCondor pool as batch campaigns.

A campaign is a self-contained directory of serialized task files that
independent condor jobs execute via ``python -m pyphi.campaign run``; results
are collected from per-task output files. :func:`prepare` writes the
directory, the user submits the generated submit file with ``condor_submit``,
and :func:`status` / :func:`collect` operate purely on the directory's
files — a task is done exactly when its output file exists and loads.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import re
import stat
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec

from pyphi import serialize
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.cost import estimate_analysis
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.models.pandas import ToPandasMixin
from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.sweep import _UNREACHABLE
from pyphi.sweep import SweepResult
from pyphi.sweep import _build_df
from pyphi.sweep import _enumerate_cells
from pyphi.sweep import _extract_row
from pyphi.sweep import _formalism_preset
from pyphi.sweep import _normalize_formalisms
from pyphi.sweep import _normalize_substrates
from pyphi.warnings import PyPhiWarning

__all__ = [
    "CESShardTask",
    "CampaignStatus",
    "CampaignTask",
    "CampaignTaskOutput",
    "CellOutput",
    "SIAShardTask",
    "ScopeReport",
    "collect",
    "prepare",
    "prepare_ces",
    "scope_report",
    "status",
]


@dataclass(frozen=True)
class CampaignTask:
    """One condor job's worth of work: the cells it owns and their context.

    ``config_overrides`` is the preparing session's configuration in override
    form (JSON-compatible); the runner installs it beneath each cell's
    formalism preset. ``compute_ref`` is a ``"module:qualname"`` reference
    used when the computation is a callable rather than ``"sia"``/``"ces"``.
    """

    task_id: int
    kind: str
    compute: str | None
    compute_ref: str | None
    config_overrides: dict[str, Any]
    cells: tuple[tuple[Any, str | None, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool


@dataclass(frozen=True)
class CellOutput:
    """One cell's outcome: ``ok`` (with the result), ``skipped``, or ``error``.

    ``aux`` carries per-entry bookkeeping some task kinds need at merge
    time (tie-set enumeration indices, the active partition scheme).
    """

    status: str
    result: Any | None
    traceback: str | None
    aux: dict[str, Any] | None = None


@dataclass(frozen=True)
class CampaignTaskOutput:
    """A task's per-cell outcomes, aligned 1:1 with the task's cells.

    ``metrics`` records what the task cost to run: ``wall_s``, ``cpu_s``,
    ``cache_hits``, ``cache_misses``, ``cache_evictions``, and for a shard
    task the ``payload_kind``, ``units``, ``memory_bytes``, and
    ``n_mechanisms`` it was packed against. These are the observed side of
    what :mod:`pyphi.cost` predicts, so a campaign's own outputs recalibrate
    :data:`pyphi.cost.SECONDS_PER_UNIT` for the hardware it ran on.
    """

    task_id: int
    pyphi_version: str
    entries: tuple[CellOutput, ...]
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class CESShardTask:
    """One shard of a scoped cause-effect computation for one system."""

    task_id: int
    kind: str
    substrate_label: Any
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    scope: Any
    config_overrides: dict[str, Any]
    formalism: str | None
    spec: Any
    ordering: str | None


@dataclass(frozen=True)
class SIAShardTask:
    """One stride of the system-partition sweep for one system."""

    task_id: int
    kind: str
    substrate_label: Any
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    config_overrides: dict[str, Any]
    formalism: str | None
    stride: tuple[int, int]


@dataclass(frozen=True)
class CampaignStatus(Displayable, ToPandasMixin):
    """A campaign's task ledger: which tasks are done, failed, or pending."""

    directory: str
    n_tasks: int
    n_cells: int
    done: tuple[int, ...]
    failed: tuple[int, ...]
    pending: tuple[int, ...]
    total_units: float

    def _pandas_record(self) -> dict:
        return {
            "directory": self.directory,
            "n_tasks": self.n_tasks,
            "n_cells": self.n_cells,
            "done": len(self.done),
            "failed": len(self.failed),
            "pending": len(self.pending),
            "total_units": self.total_units,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        rows = [
            Row("Directory", self.directory),
            Row("Tasks", str(self.n_tasks)),
            Row("Cells", str(self.n_cells)),
            Row("Done", str(len(self.done))),
            Row("Failed", str(len(self.failed))),
            Row("Pending", str(len(self.pending))),
            Row("Total work units", f"{self.total_units:.3g}"),
        ]
        return Description(
            title="CampaignStatus",
            subtitle=f"{len(self.done)}/{self.n_tasks} tasks done",
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"CampaignStatus(done={len(self.done)}/{self.n_tasks}, "
                f"failed={len(self.failed)})"
            ),
        )


def _wire_overrides() -> dict[str, Any]:
    """The active configuration as JSON-compatible override kwargs."""

    def enc_hook(x: Any) -> Any:
        return dict(x) if isinstance(x, Mapping) else str(x)

    overrides = config.snapshot().as_overrides()
    return json.loads(json.dumps(msgspec.to_builtins(overrides, enc_hook=enc_hook)))


def _resolve_compute_ref(ref: str) -> Any:
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _compute_spec(compute: Any) -> tuple[str | None, str | None]:
    """Split a compute argument into (name, importable reference)."""
    if isinstance(compute, str):
        if compute not in ("sia", "ces"):
            raise ValueError(
                f"unknown compute: {compute!r}; expected 'sia', 'ces', or a callable"
            )
        return compute, None
    ref = f"{compute.__module__}:{compute.__qualname__}"
    try:
        resolved = _resolve_compute_ref(ref)
    except (ImportError, AttributeError):
        resolved = None
    if resolved is not compute:
        raise ValueError(
            f"compute callable {compute!r} is not importable as {ref!r}; "
            "campaign computations must be module-level functions "
            "(lambdas and local functions cannot ship to jobs)"
        )
    return None, ref


def _axis_as_json(axis: Any) -> Any:
    """An axis argument in JSON form: a mode string, or a list of lists."""
    if isinstance(axis, str):
        return axis
    if isinstance(axis, tuple) and all(isinstance(x, int) for x in axis):
        return [list(axis)]
    return [list(x) for x in axis]


def _cell_weights(
    cells: list, substrate_map: dict, compute_name: str | None
) -> tuple[list[float], list[bool]]:
    """Per-cell work estimates.

    Estimates are state-independent, so they are memoized per
    (label, formalism, subset). Callable computations cannot be estimated
    and fall back to uniform weights.
    """
    memo: dict[tuple, tuple[float, bool]] = {}
    weights, capped = [], []
    for label, formalism, subset, _state in cells:
        key = (label, formalism, subset)
        if key not in memo:
            if compute_name is None:
                memo[key] = (1.0, False)
            else:
                with config.override(
                    **_formalism_preset(formalism), progress_bars=False
                ):
                    est = estimate_analysis(
                        substrate_map[label], subset=subset, compute=compute_name
                    )
                axes = (
                    est.system_partitions,
                    est.mechanisms,
                    est.purview_evaluations,
                    est.mechanism_partition_sweeps,
                )
                memo[key] = (
                    float(sum(a for a in axes if a is not None)),
                    est.capped,
                )
        weight, was_capped = memo[key]
        weights.append(weight)
        capped.append(was_capped)
    return weights, capped


def _pack(
    weights: list[float], jobs: int | None, units_per_job: float | None
) -> list[list[int]]:
    if jobs is not None and units_per_job is not None:
        raise ValueError("pass either jobs or units_per_job, not both")
    if units_per_job is not None:
        jobs = max(1, math.ceil(sum(weights) / units_per_job))
    if jobs is None:
        return [[i] for i in range(len(weights))]
    bins = cost_balanced_partition(weights, jobs)
    return [sorted(b) for b in bins]


# Task files are written zero-padded (``task-{task_id:04d}.json.gz``), and
# collection reads them back with the same padding, so every filename the
# scheduler forms from ``task_id`` must pad identically. ``{pad}`` is filled
# with HTCondor's ``$INT(task_id,%04d)``, which pads from the unpadded
# ``task_id`` column in ``remaining.txt`` (which ``status`` regenerates
# unpadded), keeping the submit file correct across resubmits. A bare
# ``$(task_id)`` here would expand to ``task-0.json.gz`` and every job would
# fail its input-file transfer.
_TASK_ID_PAD = "$INT(task_id,%04d)"
_SUBMIT_TEMPLATE = """\
universe            = container
container_image     = {container_image}
executable          = run_task.sh
arguments           = {pad}
transfer_input_files = tasks/task-{pad}.json.gz, substrates/
transfer_output_remaps = "task-{pad}.json.gz = outputs/task-{pad}.json.gz"
should_transfer_files = YES
# Keep the tasks/ and substrates/ layout on the execute node; without this
# HTCondor flattens tasks/task-XXXX.json.gz to the scratch root and
# run_task.sh's "tasks/task-$1.json.gz" cannot find it.
preserve_relative_paths = true
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus        = 1
request_memory      = $(memory)
# The granted allocation, so the shard's cache ceiling tracks the request
# rather than the value planning recorded in the task file. Editing a row in
# remaining.txt then moves both together.
environment         = "PYPHI_SHARD_MEMORY=$(memory)"
request_disk        = {request_disk}
log                 = logs/task-{pad}.log
output              = logs/task-{pad}.out
error               = logs/task-{pad}.err
queue task_id, memory from remaining.txt
"""

_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(MB|GB)\s*$", re.IGNORECASE)


def _parse_memory(value: str) -> int:
    """Parse a scheduler memory string (``"4GB"``, ``"512MB"``) to bytes."""
    match = _MEMORY_RE.match(value)
    if match is None:
        raise ValueError(
            f"cannot parse memory value {value!r}; expected e.g. '4GB' or '512MB'"
        )
    number, unit = float(match.group(1)), match.group(2).upper()
    return int(number * (1024**3 if unit == "GB" else 1024**2))


def _format_memory(n: int) -> str:
    return f"{n // 1024**2}MB"


def _remaining_lines(memory_by_task: dict[int, str]) -> str:
    return "".join(
        f"{task_id}, {memory}\n" for task_id, memory in sorted(memory_by_task.items())
    )


def _task_memory_strings(manifest: dict) -> list[str]:
    """Per-task memory request strings for either campaign kind."""
    if manifest["kind"] == "sweep_cells":
        return [manifest["request_memory"]] * len(manifest["tasks"])
    return [_format_memory(row["memory_bytes"]) for row in manifest["tasks"]]


_RUN_TASK_SH = (
    "#!/bin/bash\n"
    "set -e\n"
    "exec python -m pyphi.campaign run tasks/task-$1.json.gz"
    " --substrates substrates --outputs .\n"
)


def prepare(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    compute: Any = "sia",
    directory: Any,
    jobs: int | None = None,
    units_per_job: float | None = None,
    infeasible_threshold: float = 1e9,
    strict: bool = False,
    container_image: str = "pyphi.sif",
    request_memory: str = "4GB",
    request_disk: str = "4GB",
    seed: int | None = None,
) -> CampaignStatus:
    """Materialize a sweep as a self-contained HTCondor campaign directory.

    Enumerates exactly the cells :func:`pyphi.sweep.sweep` would run over the
    same axes, estimates each cell's workload with
    :func:`pyphi.cost.estimate_analysis` under its formalism preset, packs
    cells into cost-balanced tasks, and writes the campaign directory: one
    serialized task file per condor job, each substrate serialized once, the
    generated submit file, and a manifest recording every estimate and
    packing decision. Submit with ``condor_submit pyphi.sub`` from the
    campaign directory; monitor and collect with :func:`status` and
    :func:`collect`.

    Parameters
    ----------
    substrates
        As in :func:`pyphi.sweep.sweep`: one substrate, a sequence, or a
        ``{label: substrate}`` mapping.
    states, subsets, formalisms, compute
        As in :func:`pyphi.sweep.sweep`. A callable ``compute`` must be an
        importable module-level function.
    directory
        Target directory; created, and must not already exist.
    jobs : int, optional
        Pack cells into exactly this many cost-balanced tasks.
    units_per_job : float, optional
        Target work units per task; the task count is
        ``ceil(total / units_per_job)``. Mutually exclusive with ``jobs``;
        with neither, each cell is its own task.
    infeasible_threshold : float, optional
        A single cell whose estimate exceeds this triggers a warning naming
        the cell (or an error with ``strict``). The default marks cells that
        cannot finish in a 72-hour slot unless per-unit cost is well below a
        millisecond.
    strict : bool, optional
        Escalate admission-control warnings to errors.
    container_image, request_memory, request_disk : str, optional
        Substituted into the generated submit file.
    seed : int, optional
        Recorded in the manifest; stamped into result provenance by
        :func:`collect`.

    Returns
    -------
    CampaignStatus
        The freshly prepared ledger (all tasks pending).
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"campaign directory {directory} already exists; "
            "campaign directories are never overwritten"
        )
    compute_name, compute_ref = _compute_spec(compute)
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    formalisms_ = _normalize_formalisms(formalisms)
    cells = _enumerate_cells(labeled, states, subsets, formalisms_)
    if not cells:
        raise ValueError("the given axes enumerate no cells")
    skip_uncomputable = states == "all" or subsets == "all"

    weights, capped = _cell_weights(cells, substrate_map, compute_name)
    for cell, weight in zip(cells, weights, strict=True):
        if weight > infeasible_threshold:
            message = (
                f"cell {cell!r} estimate {weight:.3g} exceeds "
                f"infeasible_threshold {infeasible_threshold:.3g}; consider "
                "narrowing the axes or raising the threshold"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, PyPhiWarning, stacklevel=2)
    tasks = _pack(weights, jobs, units_per_job)

    directory.mkdir(parents=True)
    (directory / "outputs").mkdir()
    (directory / "logs").mkdir()
    substrates_dir = directory / "substrates"
    substrates_dir.mkdir()
    for label, substrate in labeled:
        serialize.save(substrate, substrates_dir / f"substrate-{label}.msgpack.gz")

    tasks_dir = directory / "tasks"
    tasks_dir.mkdir()
    overrides = _wire_overrides()
    for task_id, cell_indices in enumerate(tasks):
        task = CampaignTask(
            task_id=task_id,
            kind="sweep_cells",
            compute=compute_name,
            compute_ref=compute_ref,
            config_overrides=overrides,
            cells=tuple(cells[i] for i in cell_indices),
            skip_uncomputable=skip_uncomputable,
        )
        serialize.save(task, tasks_dir / f"task-{task_id:04d}.json.gz")

    manifest = {
        "kind": "sweep_cells",
        "pyphi_version": importlib.metadata.version("pyphi"),
        "created": datetime.now(UTC).isoformat(),
        "seed": seed,
        "compute": compute_name,
        "compute_ref": compute_ref,
        "axes": {
            "states": _axis_as_json(states),
            "subsets": _axis_as_json(subsets),
            "formalisms": list(formalisms_),
        },
        # Labels the ambient-config sentinel (formalism None) in collected
        # tables; the executed configuration itself travels in each task's
        # config_overrides.
        "active_version": config.formalism.iit.version,
        "substrate_labels": [label for label, _ in labeled],
        "cells": [
            [label, formalism, list(subset), list(state)]
            for label, formalism, subset, state in cells
        ],
        "weights": weights,
        "capped": capped,
        "tasks": tasks,
        "skip_uncomputable": skip_uncomputable,
        "infeasible_threshold": infeasible_threshold,
        "packing": {"jobs": jobs, "units_per_job": units_per_job},
        "request_memory": request_memory,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_campaign_scaffold(
        directory, [request_memory] * len(tasks), container_image, request_disk
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(tasks),
        n_cells=len(cells),
        done=(),
        failed=(),
        pending=tuple(range(len(tasks))),
        total_units=float(sum(weights)),
    )


def _write_campaign_scaffold(
    directory: Path,
    memory_by_task: list[str],
    container_image: str,
    request_disk: str,
) -> None:
    """Write the scheduler-facing campaign files common to every kind."""
    (directory / "remaining.txt").write_text(
        _remaining_lines(dict(enumerate(memory_by_task)))
    )
    run_task_sh = directory / "run_task.sh"
    run_task_sh.write_text(_RUN_TASK_SH)
    run_task_sh.chmod(
        run_task_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    (directory / "pyphi.sub").write_text(
        _SUBMIT_TEMPLATE.format(
            container_image=container_image,
            request_disk=request_disk,
            pad=_TASK_ID_PAD,
        )
    )


def prepare_ces(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    scope: Any = None,
    directory: Any,
    units_per_job: float,
    limit: int = 100_000_000,
    workloads: dict[tuple[int, ...], Any] | None = None,
    sia: Any = None,
    resolution_state: Any = None,
    ordering: str | None = None,
    infeasible_threshold: float = 1e9,
    strict: bool = False,
    container_image: str = "pyphi.sif",
    request_memory: str = "4GB",
    request_disk: str = "4GB",
    seed: int | None = None,
) -> CampaignStatus:
    """Materialize a scoped CES sweep as a campaign.

    Enumerates exactly the cells :func:`pyphi.sweep.sweep` would run over
    the same axes, all sharing one scope, and plans shards for each cell's
    scoped distinction computation (and, unless a precomputed ``sia`` or
    explicit ``resolution_state`` is given, for its system irreducibility
    analysis), descending mechanism → purview-range → partition-stride
    only where ``units_per_job`` requires. The shard plan depends only on
    the substrate, subset, and formalism — not the state — so cells that
    differ only by state share one planning pass and replicate its shard
    tasks. Shards are independent condor jobs on the standard campaign
    scaffold; collection merges them exactly (tie sets preserved) and
    assembles each cell's cause-effect structure through the standard
    analysis path.

    Parameters
    ----------
    substrates
        As in :func:`pyphi.sweep.sweep`: one substrate, a sequence, or a
        ``{label: substrate}`` mapping.
    states, subsets, formalisms
        As in :func:`pyphi.sweep.sweep`; scalars are accepted anywhere.
    scope : CESScope, optional
        One feasibility surface shared by every cell, resolved per
        substrate; ``None`` is the unconstrained scope.
    directory
        Target campaign directory; created, and must not already exist.
    units_per_job : float
        Target work units per shard — the planning ladder's budget.
    limit : int, optional
        Work budget for each planning walk. The walk raises
        :class:`ValueError` past the limit — the workload is then too
        large to plan; narrow the scope or raise the limit.
    workloads : dict, optional
        Per-mechanism workloads to plan against, replacing the counting
        walk — a :func:`pyphi.cost.mechanism_workloads` mapping, or one
        derived from measured runtimes when the caller knows the cost
        better than the analytic model does. Applies to a campaign with a
        single planning group (one substrate, formalism, and subset);
        :class:`ValueError` otherwise, since a group's workloads are
        specific to its substrate and subset.
    sia : optional
        A precomputed system irreducibility analysis; suppresses SIA
        shards and is used at collection. Single-cell campaigns only.
    resolution_state : optional
        Explicit congruence-resolution states; suppresses SIA shards, and
        the collected structures then carry no Φₛ. One specification (the
        result of
        :func:`pyphi.formalism.iit4.system_intrinsic_information`) for a
        single-cell campaign; for multi-cell campaigns, a mapping keyed by
        the full ``(label, formalism, subset, state)`` cell tuples, a
        mapping keyed by state alone (when the other axes are singletons),
        or a callable ``cell -> specification``. Every cell must resolve;
        values are validated here rather than at collect time.
    ordering : {"bottleneck_first", None}, optional
        Reorder each partition-stride shard's slice so likely-reducible
        partitions are evaluated first (sparse substrates short-circuit
        sooner). Never affects results.
    infeasible_threshold : float, optional
        A shard whose estimate exceeds this triggers a warning (or an
        error with ``strict``).
    strict : bool, optional
        Escalate admission-control warnings to errors.
    container_image, request_disk : str, optional
        Substituted into the generated submit file.
    request_memory : str, optional
        Minimum per-shard memory request (the floor). Every shard requests
        the greater of this and its estimated peak, so a large floor
        disables memory stratification.
    seed : int, optional
        Recorded in the manifest; stamped into result provenance by
        :func:`collect`.

    Returns
    -------
    CampaignStatus
        The freshly prepared ledger (all tasks pending).
    """
    from pyphi.campaign import shards as _shards
    from pyphi.campaign.scope import CESScope
    from pyphi.campaign.scope import resolve_scope
    from pyphi.cost import mechanism_workloads
    from pyphi.system import System

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"campaign directory {directory} already exists; "
            "campaign directories are never overwritten"
        )
    if sia is not None and resolution_state is not None:
        raise ValueError("pass either sia or resolution_state, not both")
    formalisms_ = _normalize_formalisms(formalisms)
    for name in formalisms_:
        if name is not None and name not in presets.by_name:
            raise ValueError(f"unknown formalism {name!r}")
    scope = scope if scope is not None else CESScope()
    memory_floor = _parse_memory(request_memory)
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    cells = _enumerate_cells(labeled, states, subsets, formalisms_)
    if not cells:
        raise ValueError("the given axes enumerate no cells")
    # As in the sweep: cells enumerated via "all" skip dynamically
    # unreachable states; explicit cells fail loud.
    skip_uncomputable = states == "all" or subsets == "all"
    computable: list = []
    skipped_cells: list = []
    for cell in cells:
        label, formalism_, subset, state = cell
        with config.override(**_formalism_preset(formalism_), progress_bars=False):
            try:
                System.from_substrate(substrate_map[label], tuple(state), subset)
            except _UNREACHABLE:
                if not skip_uncomputable:
                    raise
                skipped_cells.append(cell)
                continue
        computable.append(cell)
    cells = computable
    if not cells:
        raise ValueError("every enumerated cell has an unreachable state")
    if sia is not None and len(cells) > 1:
        raise ValueError("sia applies only to single-cell campaigns")
    resolution_states = _normalize_resolution_states(resolution_state, cells)

    # Plan once per (label, formalism, subset) group; the shard plan is
    # state-independent, so states replicate tasks, not planning.
    group_keys: list[tuple[Any, str, tuple]] = []
    group_data: dict[tuple[Any, str, tuple], dict] = {}
    if workloads is not None and len({(c[0], c[1], c[2]) for c in cells}) > 1:
        raise ValueError(
            "workloads applies to a single planning group; this campaign has "
            "several (substrate, formalism, subset) groups, whose workloads "
            "differ"
        )
    for label, formalism_, subset, state in cells:
        key = (label, formalism_, subset)
        if key in group_data:
            continue
        group_keys.append(key)
        with config.override(**presets.by_name[formalism_], progress_bars=False):
            # The plan is state-independent; the group's first cell state
            # stands in for construction (and is validated here).
            system = System.from_substrate(substrate_map[label], tuple(state), subset)
            resolved = resolve_scope(scope, system.node_labels)
            group_workloads = (
                workloads
                if workloads is not None
                else mechanism_workloads(
                    substrate_map[label],
                    subset=system.node_indices,
                    scope=resolved,
                    limit=limit,
                )
            )
            ces_specs = _shards.plan_ces_shards(
                system,
                resolved,
                units_per_job,
                workloads=group_workloads,
                memory_floor_bytes=memory_floor,
            )
            if not any(s.mechanisms or s.mechanism for s in ces_specs):
                raise ValueError("the scope admits zero mechanisms")
            sia_specs = (
                _shards.plan_sia_shards(
                    system, units_per_job, memory_floor_bytes=memory_floor
                )
                if sia is None and resolution_states is None
                else []
            )
            group_data[key] = {
                "resolved": resolved,
                "workloads": group_workloads,
                "ces_specs": ces_specs,
                "sia_specs": sia_specs,
                "subset": tuple(system.node_indices),
                "partition_scheme": config.formalism.iit.system_partition_scheme,
                "mechanism_partition_scheme": (
                    config.formalism.iit.mechanism_partition_scheme
                ),
            }

    for data in group_data.values():
        for spec in data["ces_specs"] + data["sia_specs"]:
            if spec.units > infeasible_threshold:
                message = (
                    f"shard {spec!r} estimate {spec.units:.3g} exceeds "
                    f"infeasible_threshold {infeasible_threshold:.3g}"
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, PyPhiWarning, stacklevel=2)

    directory.mkdir(parents=True)
    (directory / "outputs").mkdir()
    (directory / "logs").mkdir()
    substrates_dir = directory / "substrates"
    substrates_dir.mkdir()
    for label, substrate in labeled:
        serialize.save(substrate, substrates_dir / f"substrate-{label}.msgpack.gz")
    serialize.save(scope, directory / "scope.json.gz")
    if sia is not None:
        serialize.save(sia, directory / "sia.json.gz")
    if resolution_states is not None:
        for cell_index, cell in enumerate(cells):
            serialize.save(
                resolution_states[_cell_key(cell)],
                directory / f"resolution_state-{cell_index:04d}.json.gz",
            )

    tasks_dir = directory / "tasks"
    tasks_dir.mkdir()
    overrides = _wire_overrides()
    task_rows: list[dict] = []
    task_id = 0
    for cell_index, (label, formalism_, subset, state) in enumerate(cells):
        data = group_data[(label, formalism_, subset)]
        for spec in data["ces_specs"]:
            shard_task = CESShardTask(
                task_id=task_id,
                kind="ces_shard",
                substrate_label=label,
                state=tuple(state),
                subset=data["subset"],
                scope=data["resolved"],
                config_overrides=overrides,
                formalism=formalism_,
                spec=spec,
                ordering=ordering,
            )
            serialize.save(shard_task, tasks_dir / f"task-{task_id:04d}.json.gz")
            task_rows.append(
                {
                    "task_id": task_id,
                    "kind": "ces_shard",
                    "units": spec.units,
                    "memory_bytes": spec.memory_bytes,
                    "cell": cell_index,
                }
            )
            task_id += 1
        for spec in data["sia_specs"]:
            assert spec.stride is not None, "SIA shards are always strides"
            sia_task = SIAShardTask(
                task_id=task_id,
                kind="sia_shard",
                substrate_label=label,
                state=tuple(state),
                subset=data["subset"],
                config_overrides=overrides,
                formalism=formalism_,
                stride=spec.stride,
            )
            serialize.save(sia_task, tasks_dir / f"task-{task_id:04d}.json.gz")
            task_rows.append(
                {
                    "task_id": task_id,
                    "kind": "sia_shard",
                    "units": spec.units,
                    "memory_bytes": spec.memory_bytes,
                    "cell": cell_index,
                }
            )
            task_id += 1

    sia_mode = (
        "precomputed"
        if sia is not None
        else "none"
        if resolution_states is not None
        else "shards"
    )
    manifest = {
        "kind": "ces",
        "pyphi_version": importlib.metadata.version("pyphi"),
        "created": datetime.now(UTC).isoformat(),
        "active_version": config.formalism.iit.version,
        "seed": seed,
        "sia_mode": sia_mode,
        "ordering": ordering,
        "cells": [
            [
                label,
                formalism_,
                list(group_data[(label, formalism_, subset)]["subset"]),
                list(state),
            ]
            for label, formalism_, subset, state in cells
        ],
        "skipped_cells": [
            [label, formalism_, list(subset), list(state)]
            for label, formalism_, subset, state in skipped_cells
        ],
        "groups": [
            {
                "label": label,
                "formalism": formalism_,
                "subset": list(group_data[key]["subset"]),
                "partition_scheme": group_data[key]["partition_scheme"],
                "mechanism_partition_scheme": group_data[key][
                    "mechanism_partition_scheme"
                ],
                "mechanism_workloads": {
                    ",".join(map(str, mechanism)): workload.units
                    for mechanism, workload in group_data[key]["workloads"].items()
                },
            }
            for key in group_keys
            for label, formalism_, _subset in [key]
        ],
        "tasks": task_rows,
        "units_per_job": units_per_job,
        "infeasible_threshold": infeasible_threshold,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_campaign_scaffold(
        directory,
        [_format_memory(row["memory_bytes"]) for row in task_rows],
        container_image,
        request_disk,
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(task_rows),
        n_cells=len(cells),
        done=(),
        failed=(),
        pending=tuple(range(len(task_rows))),
        total_units=float(sum(row["units"] for row in task_rows)),
    )


def _load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _cell_key(cell: Any) -> tuple:
    """A cell in canonical key form: subset and state as tuples."""
    label, formalism_, subset, state = cell
    return (label, formalism_, tuple(subset), tuple(state))


def _validate_resolution_spec(spec: Any, cell: tuple) -> None:
    from pyphi.direction import Direction

    try:
        cause = spec[Direction.CAUSE]
        effect = spec[Direction.EFFECT]
    except Exception:
        cause = effect = None
    if not (hasattr(cause, "state") and hasattr(effect, "state")):
        raise TypeError(
            f"resolution_state for cell {cell!r} must be a system state "
            "specification carrying CAUSE and EFFECT state specifications — "
            "the result of pyphi.formalism.iit4.system_intrinsic_information — "
            f"not {type(spec).__name__}"
        )


def _normalize_resolution_states(resolution_state: Any, cells: list) -> dict | None:
    """Resolve ``resolution_state`` to one validated specification per cell.

    Accepts a single specification (single-cell campaigns only), a mapping
    keyed by the full ``(label, formalism, subset, state)`` cell tuples
    ``_enumerate_cells`` produces, a mapping keyed by state alone (only when
    the other three axes are singletons), or a callable
    ``cell -> specification``. Every enumerated cell must resolve to a
    specification; each is type-validated here, at preparation time, rather
    than failing deep inside congruence resolution at collect time.
    """
    if resolution_state is None:
        return None
    keys = [_cell_key(cell) for cell in cells]
    if isinstance(resolution_state, Mapping):
        state_keyed = all(
            isinstance(k, tuple) and all(isinstance(x, int) for x in k)
            for k in resolution_state
        )
        if state_keyed:
            for axis, values in (
                ("substrate", {k[0] for k in keys}),
                ("formalism", {k[1] for k in keys}),
                ("subset", {k[2] for k in keys}),
            ):
                if len(values) > 1:
                    raise ValueError(
                        f"a state-keyed resolution_state mapping is ambiguous "
                        f"when the {axis} axis is not a singleton; key the "
                        "mapping by full cell tuples "
                        "(label, formalism, subset, state) instead"
                    )
            label0, formalism0, subset0, _ = keys[0]
            lookup = {
                (label0, formalism0, subset0, tuple(state)): spec
                for state, spec in resolution_state.items()
            }
        else:
            lookup = {_cell_key(k): v for k, v in resolution_state.items()}
    elif callable(resolution_state):
        lookup = {}
        for key in keys:
            try:
                lookup[key] = resolution_state(key)
            except KeyError:
                raise ValueError(
                    f"resolution_state has no entry for cell {key!r}"
                ) from None
    else:
        if len(cells) > 1:
            raise ValueError(
                "a single resolution_state applies only to single-cell "
                "campaigns; pass a mapping keyed by cell tuple or state, "
                "or a callable cell -> specification"
            )
        lookup = {keys[0]: resolution_state}

    resolved: dict[tuple, Any] = {}
    for key in keys:
        if key not in lookup:
            raise ValueError(f"resolution_state has no entry for cell {key!r}")
        spec = lookup[key]
        _validate_resolution_spec(spec, key)
        resolved[key] = spec
    return resolved


def _label_formalism(cells: Any, manifest: dict) -> list:
    """Replace the ambient-config sentinel (None) with the version name the
    campaign was prepared under, for reporting only."""
    version = manifest.get("active_version")
    return [
        (label, version if formalism is None else formalism, subset, state)
        for label, formalism, subset, state in cells
    ]


def _manifest_cells(manifest: dict) -> list[tuple[Any, str | None, tuple, tuple]]:
    return [
        (label, formalism, tuple(subset), tuple(state))
        for label, formalism, subset, state in manifest["cells"]
    ]


def status(directory: Any) -> CampaignStatus:
    """Classify every task from the output files and refresh ``remaining.txt``.

    A task is done exactly when its output file exists, loads, and every
    entry is ``ok`` or ``skipped``; it is failed when the output loads with
    an ``error`` entry or does not load; otherwise it is pending. Pending
    and failed task ids are rewritten to ``remaining.txt``, so resubmission
    is running ``condor_submit pyphi.sub`` again.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    done, failed, pending = [], [], []
    for task_id in range(len(manifest["tasks"])):
        path = directory / "outputs" / f"task-{task_id:04d}.json.gz"
        if not path.exists():
            pending.append(task_id)
            continue
        try:
            output = serialize.load(path)
        except Exception:  # an unloadable output is a failed task
            failed.append(task_id)
            continue
        if any(entry.status == "error" for entry in output.entries):
            failed.append(task_id)
        else:
            done.append(task_id)
    memory = _task_memory_strings(manifest)
    (directory / "remaining.txt").write_text(
        _remaining_lines(
            {task_id: memory[task_id] for task_id in sorted(pending + failed)}
        )
    )
    if "weights" in manifest:
        total_units = float(sum(manifest["weights"]))
    else:
        total_units = float(sum(row["units"] for row in manifest["tasks"]))
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(manifest["tasks"]),
        n_cells=len(manifest.get("cells", manifest["tasks"])),
        done=tuple(done),
        failed=tuple(failed),
        pending=tuple(pending),
        total_units=total_units,
    )


def collect(
    directory: Any,
    partial: bool = False,
    sia: Any = None,
    resolution_state: Any = None,
) -> Any:
    """Reassemble a campaign's outputs into its result.

    Sweep campaigns return the exact local-sweep :class:`SweepResult`; CES
    campaigns return the assembled
    :class:`~pyphi.models.ces.CauseEffectStructure` (merging shard tie
    sets exactly, resolving congruence, and computing relations through
    the standard analysis path), writing a scope report alongside. With
    missing or failed tasks the default is to raise with a per-task
    summary; ``partial=True`` instead warns and returns the result built
    from the completed tasks.

    Parameters
    ----------
    directory
        A campaign directory whose tasks have been executed.
    partial : bool, optional
        Return the partial result instead of raising on incomplete tasks.
    sia : optional
        For CES campaigns: a system irreducibility analysis to use at
        assembly, overriding whatever the campaign planned.
    resolution_state : optional
        For CES campaigns without any SIA: explicit congruence-resolution
        states, in any of the forms :func:`prepare_ces` accepts (a single
        specification for single-cell campaigns, a cell- or state-keyed
        mapping, or a callable). Overrides the states stored at
        preparation.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    if manifest["kind"] != "sweep_cells":
        return _collect_ces(directory, manifest, partial, sia, resolution_state)
    if sia is not None or resolution_state is not None:
        raise ValueError("sia/resolution_state apply only to CES campaigns")
    return _collect_sweep(directory, manifest, partial)


def _collect_sweep(directory: Path, manifest: dict, partial: bool) -> SweepResult:
    st = status(directory)
    incomplete = sorted(set(st.failed) | set(st.pending))
    if incomplete:
        summary = (
            f"{len(incomplete)} of {st.n_tasks} tasks incomplete "
            f"(failed: {list(st.failed)}, pending: {list(st.pending)}); "
            "resubmit with condor_submit pyphi.sub"
        )
        if not partial:
            raise RuntimeError(summary)
        warnings.warn(summary, PyPhiWarning, stacklevel=2)
    incomplete_set = set(incomplete)

    cells = _manifest_cells(manifest)
    compute = manifest["compute"] if manifest["compute"] is not None else "callable"
    by_index: dict[int, tuple[str, Any]] = {}
    for task_id, cell_indices in enumerate(manifest["tasks"]):
        if task_id in incomplete_set:
            continue
        output = serialize.load(directory / "outputs" / f"task-{task_id:04d}.json.gz")
        for cell_index, entry in zip(cell_indices, output.entries, strict=True):
            by_index[cell_index] = (entry.status, entry.result)

    keys, raw, skipped = [], [], []
    for cell_index in sorted(by_index):
        cell = cells[cell_index]
        entry_status, result = by_index[cell_index]
        if entry_status == "skipped":
            skipped.append(cell)
        else:
            keys.append(cell)
            raw.append(result)

    if manifest["seed"] is not None:
        for result in raw:
            with_provenance = getattr(result, "with_provenance", None)
            if with_provenance is not None:
                with_provenance(seed=manifest["seed"])

    rows = [_extract_row(result, compute) for result in raw]
    df = _build_df(
        _label_formalism(keys, manifest), rows, _label_formalism(cells, manifest)
    )
    return SweepResult(df=df, results=raw, skipped=_label_formalism(skipped, manifest))


@dataclass(frozen=True)
class ScopeReport(Displayable, ToPandasMixin):
    """What a scoped campaign computed, excluded, and certifies.

    ``sum_phi_r_lower`` is the Σφ_r of the computed relations — an exact
    lower bound for the full structure, since partial structures are exact
    substructures. ``sum_phi_r_upper`` and ``big_phi_upper`` are the
    measured certificates from :mod:`pyphi.formalism.iit4.bounds`.
    """

    mechanisms_computed: int
    mechanisms_admitted: int
    mechanisms_possible: int
    missing_groups: tuple[str, ...]
    sum_phi_r_lower: float
    sum_phi_r_upper: float | None
    big_phi_upper: float | None
    sia_mode: str

    def _pandas_record(self) -> dict:
        return {
            "mechanisms_computed": self.mechanisms_computed,
            "mechanisms_admitted": self.mechanisms_admitted,
            "mechanisms_possible": self.mechanisms_possible,
            "missing_groups": len(self.missing_groups),
            "sum_phi_r_lower": self.sum_phi_r_lower,
            "sum_phi_r_upper": self.sum_phi_r_upper,
            "big_phi_upper": self.big_phi_upper,
            "sia_mode": self.sia_mode,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        rows = [
            Row("Mechanisms computed", str(self.mechanisms_computed)),
            Row("Mechanisms admitted by scope", str(self.mechanisms_admitted)),
            Row("Mechanisms possible", str(self.mechanisms_possible)),
            Row("Missing groups", str(len(self.missing_groups))),
            Row("Σφ_r lower bound", f"{self.sum_phi_r_lower:.6g}"),
            Row(
                "Σφ_r upper bound",
                "—" if self.sum_phi_r_upper is None else f"{self.sum_phi_r_upper:.6g}",
            ),
            Row(
                "Φ upper bound",
                "—" if self.big_phi_upper is None else f"{self.big_phi_upper:.6g}",
            ),
            Row("SIA mode", self.sia_mode),
        ]
        return Description(
            title="ScopeReport",
            subtitle=(
                f"{self.mechanisms_computed}/{self.mechanisms_admitted} admitted "
                "mechanisms computed"
            ),
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"ScopeReport(computed={self.mechanisms_computed}, "
                f"admitted={self.mechanisms_admitted})"
            ),
        )


def scope_report(directory: Any) -> Any:
    """Read the scope report(s) a CES campaign's collection wrote.

    Returns one :class:`ScopeReport` for a single-cell campaign, or a dict
    keyed by ``(label, formalism, subset, state)`` cell tuples for a
    multi-cell one.
    """
    path = Path(directory) / "scope_report.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; collect the campaign first")
    data = json.loads(path.read_text())
    if "cells" not in data:
        data["missing_groups"] = tuple(data["missing_groups"])
        return ScopeReport(**data)
    reports = {}
    for entry in data["cells"]:
        label, formalism_, subset, state = entry["cell"]
        report = dict(entry["report"])
        report["missing_groups"] = tuple(report["missing_groups"])
        reports[(label, formalism_, tuple(subset), tuple(state))] = ScopeReport(**report)
    return reports


def _group_name(task: Any) -> str:
    if getattr(task, "kind", None) == "sia_shard":
        return "sia"
    spec = task.spec
    if spec.payload_kind == "mechanisms":
        return f"mechanisms:{spec.mechanisms}"
    if spec.payload_kind == "purview_range":
        return f"range:{tuple(spec.mechanism)}:{spec.direction}"
    return f"stride:{tuple(spec.mechanism)}:{spec.direction}:{tuple(spec.purview)}"


_RESOLUTION_STATE_WARN_UNITS = 12
"""System size above which collect warns before computing the
congruence-resolution state itself: ``system_intrinsic_information``
enumerates every system state against every mechanism state (~4ⁿ), taking
hours beyond ~12 units and raising as infeasible beyond ~16."""


def _assemble_without_sia(system: Any, distinctions: Any, resolution_state: Any):
    """Assemble a structure whose congruence state comes without any Φₛ."""
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.relations import relations as compute_relations

    if resolution_state is None:
        n = len(system.node_indices)
        if n > _RESOLUTION_STATE_WARN_UNITS:
            warnings.warn(
                f"no resolution_state was given, so collect will compute "
                f"system_intrinsic_information over {n} units; its cost "
                f"grows ~4^n (hours beyond ~{_RESOLUTION_STATE_WARN_UNITS} "
                f"units) and may dwarf the collected computation. Pass "
                f"resolution_state (or a precomputed sia) to skip this.",
                PyPhiWarning,
                stacklevel=2,
            )
        resolution_state = system_intrinsic_information(
            system,
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure  # pyright: ignore[reportAttributeAccessIssue]
            ),
        )
    resolved = distinctions.resolve_congruence(resolution_state)
    return CauseEffectStructure(
        sia=NullSystemIrreducibilityAnalysis(
            system_state=resolution_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
        ),
        distinctions=resolved,
        relations=compute_relations(resolved),
    )


def _build_scope_report(
    group: dict,
    system: Any,
    result: Any,
    missing_groups: set,
    sia_mode: str,
) -> ScopeReport:
    from pyphi.formalism.iit4 import bounds

    n = len(system.node_indices)
    resolved = result.distinctions
    if len(resolved):
        upper_r = float(bounds.sum_phi_relations_measured_bound(resolved).value)
        upper_phi = float(bounds.big_phi_measured_bound(resolved).value)
    else:
        upper_r = None
        upper_phi = None
    return ScopeReport(
        mechanisms_computed=len(resolved),
        mechanisms_admitted=len(group["mechanism_workloads"]),
        mechanisms_possible=2**n - 1,
        missing_groups=tuple(sorted(missing_groups)),
        sum_phi_r_lower=float(result.relations.sum_phi()),
        sum_phi_r_upper=upper_r,
        big_phi_upper=upper_phi,
        sia_mode=sia_mode,
    )


def _merge_cell(
    directory: Path,
    manifest: dict,
    group: dict,
    rows: list[dict],
    incomplete: set[int],
    system: Any,
    scope: Any,
    sia_override: Any,
    resolution_state_override: Any,
) -> tuple[Any, ScopeReport]:
    """Merge one cell's shard outputs into its structure and report."""
    from pyphi.campaign import merge as _merge
    from pyphi.direction import Direction
    from pyphi.models.distinctions import UnresolvedDistinctions

    # Group loaded outputs by reconstruction target.
    whole_distinctions: dict[tuple, Any] = {}
    purview_rias: dict[tuple, dict[tuple, Any]] = {}
    stride_entries: dict[tuple, list[tuple[Any, dict]]] = {}
    sia_entries: list[tuple[Any, dict]] = []
    n_sia_outputs = 0
    missing_groups: set[str] = set()

    expected_schemes = {
        "sia_shard": group["partition_scheme"],
        "ces_shard": group["mechanism_partition_scheme"],
    }
    for row in rows:
        task_id = row["task_id"]
        task = serialize.load(directory / "tasks" / f"task-{task_id:04d}.json.gz")
        if task_id in incomplete:
            missing_groups.add(_group_name(task))
            continue
        output = serialize.load(directory / "outputs" / f"task-{task_id:04d}.json.gz")
        # Stride semantics depend on the enumeration order, a property
        # of the PyPhi version and partition scheme; refuse to merge
        # outputs produced under a different one.
        if output.pyphi_version != manifest["pyphi_version"]:
            raise RuntimeError(
                f"task {task_id} was run under pyphi "
                f"{output.pyphi_version} but the campaign was prepared "
                f"under {manifest['pyphi_version']}; re-run the task"
            )
        for entry in output.entries:
            if entry.aux is not None and "scheme" in entry.aux:
                expected = expected_schemes[row["kind"]]
                if entry.aux["scheme"] != expected:
                    raise RuntimeError(
                        f"task {task_id} ran under partition scheme "
                        f"{entry.aux['scheme']!r} but the manifest "
                        f"records {expected!r}; re-run the task"
                    )
        if row["kind"] == "sia_shard":
            # One entry per (cause, effect) specified-state pair when the
            # cell's states tie; a single entry otherwise.
            n_sia_outputs += 1
            sia_entries.extend((entry.result, entry.aux) for entry in output.entries)
            continue
        spec = task.spec
        if spec.payload_kind == "mechanisms":
            for mechanism, entry in zip(spec.mechanisms, output.entries, strict=True):
                whole_distinctions[tuple(mechanism)] = entry.result
        elif spec.payload_kind == "purview_range":
            bucket = purview_rias.setdefault((tuple(spec.mechanism), spec.direction), {})
            for purview, entry in zip(spec.purviews, output.entries, strict=True):
                bucket[tuple(purview)] = entry.result
        elif spec.payload_kind == "partition_stride":
            # One entry per specified-state pin (a single entry under
            # pin-less formalisms).
            bucket = stride_entries.setdefault(
                (tuple(spec.mechanism), spec.direction, tuple(spec.purview)),
                [],
            )
            for entry in output.entries:
                bucket.append((entry.result, entry.aux))

    # Bottom-up: strides -> per-purview RIAs.
    for (mechanism, direction, purview), entries in stride_entries.items():
        if f"stride:{mechanism}:{direction}:{purview}" in missing_groups:
            continue
        merged = _merge.merge_stride_rias(entries)
        purview_rias.setdefault((mechanism, direction), {})[purview] = merged

    # Per-purview RIAs -> MICE -> distinctions for split mechanisms.
    split_mechanisms: dict[tuple, dict[str, Any]] = {}
    for (mechanism, direction), by_purview in purview_rias.items():
        if f"range:{mechanism}:{direction}" in missing_groups:
            continue
        dir_ = Direction[direction]
        axis = scope.purview_axis(dir_, tuple(mechanism))
        canonical = list(
            axis.select(
                system.potential_purviews(dir_, mechanism, max_order=axis.order_bound())
            )
        )
        if set(map(tuple, canonical)) - set(by_purview):
            missing_groups.add(f"range:{mechanism}:{direction}")
            continue
        mice = _merge.merge_purview_rias(
            dir_, [by_purview[tuple(p)] for p in canonical], canonical
        )
        split_mechanisms.setdefault(mechanism, {})[direction] = mice
    for mechanism, mice_by_dir in split_mechanisms.items():
        if "CAUSE" in mice_by_dir and "EFFECT" in mice_by_dir:
            whole_distinctions[mechanism] = _merge.build_distinction(
                mechanism, mice_by_dir["CAUSE"], mice_by_dir["EFFECT"]
            )
        else:
            missing_groups.add(f"mechanism:{mechanism}")

    distinctions = UnresolvedDistinctions(
        tuple(d for d in whole_distinctions.values() if d)
    )

    # SIA per mode.
    sia_mode = manifest["sia_mode"]
    sia = sia_override
    if sia is None and (directory / "sia.json.gz").exists():
        sia = serialize.load(directory / "sia.json.gz")
    resolution_state = resolution_state_override
    if sia is None and sia_mode == "shards":
        n_sia_tasks = sum(1 for row in rows if row["kind"] == "sia_shard")
        if sia_entries and n_sia_outputs == n_sia_tasks:
            sia = _merge.merge_sia_strides(sia_entries, system=system)
        else:
            missing_groups.add("sia")

    if sia is not None:
        result = system.ces(sia=sia, distinctions=distinctions)
    else:
        result = _assemble_without_sia(system, distinctions, resolution_state)

    report = _build_scope_report(group, system, result, missing_groups, sia_mode)
    return result, report


def _collect_ces(
    directory: Path,
    manifest: dict,
    partial: bool,
    sia_override: Any,
    resolution_state_override: Any,
) -> Any:
    from pyphi.campaign.scope import resolve_scope
    from pyphi.system import System

    st = status(directory)
    incomplete = set(st.failed) | set(st.pending)
    if incomplete:
        summary = (
            f"{len(incomplete)} of {st.n_tasks} tasks incomplete "
            f"(failed: {sorted(st.failed)}, pending: {sorted(st.pending)}); "
            "resubmit with condor_submit pyphi.sub"
        )
        if not partial:
            raise RuntimeError(summary)
        warnings.warn(summary, PyPhiWarning, stacklevel=3)

    cells = _manifest_cells(manifest)
    multi = len(cells) > 1
    if multi and sia_override is not None:
        raise ValueError("sia applies only to single-cell campaigns")
    resolution_states = _normalize_resolution_states(resolution_state_override, cells)
    if resolution_states is None:
        loaded = {
            _cell_key(cell): serialize.load(path)
            for cell_index, cell in enumerate(cells)
            if (
                path := directory / f"resolution_state-{cell_index:04d}.json.gz"
            ).exists()
        }
        resolution_states = loaded or None
    groups = {
        (g["label"], g["formalism"], tuple(g["subset"])): g for g in manifest["groups"]
    }
    rows_by_cell: dict[int, list[dict]] = {}
    for row in manifest["tasks"]:
        rows_by_cell.setdefault(row["cell"], []).append(row)

    # Substrate labels appear in filenames, so a bare substrate's label 0
    # round-trips as the string "0".
    substrates = {
        path.name.removeprefix("substrate-").removesuffix(".msgpack.gz"): serialize.load(
            path
        )
        for path in (directory / "substrates").glob("substrate-*.msgpack.gz")
    }
    user_scope = serialize.load(directory / "scope.json.gz")

    structures: list[Any] = []
    reports: list[tuple[tuple, ScopeReport]] = []
    for cell_index, cell in enumerate(cells):
        label, formalism_, subset, state = cell
        substrate = substrates[str(label)]
        group = groups[(label, formalism_, tuple(subset))]
        with config.override(
            **_formalism_preset(formalism_), parallel=False, progress_bars=False
        ):
            system = System(substrate, tuple(state), node_indices=tuple(subset))
            scope = resolve_scope(user_scope, system.node_labels)
            structure, report = _merge_cell(
                directory,
                manifest,
                group,
                rows_by_cell.get(cell_index, []),
                incomplete,
                system,
                scope,
                sia_override,
                resolution_states.get(_cell_key(cell))
                if resolution_states is not None
                else None,
            )
        structures.append(structure)
        reports.append((cell, report))
        with_provenance = getattr(structure, "with_provenance", None)
        if with_provenance is not None:
            note = json.dumps(
                {
                    "campaign": str(directory),
                    "cell": [label, formalism_, list(subset), list(state)],
                    "scope_report": dataclasses.asdict(report),
                }
            )
            with_provenance(note=note, seed=manifest["seed"])

    if not multi:
        (directory / "scope_report.json").write_text(
            json.dumps(dataclasses.asdict(reports[0][1]), indent=2)
        )
        return structures[0]
    (directory / "scope_report.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell": [cell[0], cell[1], list(cell[2]), list(cell[3])],
                        "report": dataclasses.asdict(report),
                    }
                    for cell, report in reports
                ]
            },
            indent=2,
        )
    )
    rows = [_extract_row(s, "ces") for s in structures]
    labeled_cells = _label_formalism(cells, manifest)
    df = _build_df(labeled_cells, rows, labeled_cells)
    skipped = _label_formalism(
        [
            (label, formalism_, tuple(subset), tuple(state))
            for label, formalism_, subset, state in manifest.get("skipped_cells", [])
        ],
        manifest,
    )
    return SweepResult(df=df, results=structures, skipped=skipped)
