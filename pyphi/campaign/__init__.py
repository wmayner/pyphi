"""Distribute PyPhi computations across an HTCondor pool as batch campaigns.

A campaign is a self-contained directory of serialized task files that
independent condor jobs execute via ``python -m pyphi.campaign run``; results
are collected from per-task output files. :func:`prepare` writes the
directory, the user submits the generated submit file with ``condor_submit``,
and :func:`status` / :func:`collect` operate purely on the directory's
files — a task is done exactly when its output file exists and loads.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
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
from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.sweep import SweepResult
from pyphi.sweep import _build_df
from pyphi.sweep import _enumerate_cells
from pyphi.sweep import _extract_row
from pyphi.sweep import _normalize_formalisms
from pyphi.sweep import _normalize_substrates
from pyphi.warnings import PyPhiWarning

__all__ = [
    "CampaignStatus",
    "CampaignTask",
    "CampaignTaskOutput",
    "CellOutput",
    "collect",
    "prepare",
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
    cells: tuple[tuple[Any, str, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool


@dataclass(frozen=True)
class CellOutput:
    """One cell's outcome: ``ok`` (with the result), ``skipped``, or ``error``."""

    status: str
    result: Any | None
    traceback: str | None


@dataclass(frozen=True)
class CampaignTaskOutput:
    """A task's per-cell outcomes, aligned 1:1 with the task's cells."""

    task_id: int
    pyphi_version: str
    entries: tuple[CellOutput, ...]


@dataclass(frozen=True)
class CampaignStatus(Displayable):
    """A campaign's task ledger: which tasks are done, failed, or pending."""

    directory: str
    n_tasks: int
    n_cells: int
    done: tuple[int, ...]
    failed: tuple[int, ...]
    pending: tuple[int, ...]
    total_units: float

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
                with config.override(**presets.by_name[formalism], progress_bars=False):
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


_SUBMIT_TEMPLATE = """\
universe            = container
container_image     = {container_image}
executable          = run_task.sh
arguments           = $(task_id)
transfer_input_files = tasks/task-$(task_id).json.gz, substrates/
transfer_output_remaps = "task-$(task_id).json.gz = outputs/task-$(task_id).json.gz"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus        = 1
request_memory      = {request_memory}
request_disk        = {request_disk}
log                 = logs/task-$(task_id).log
output              = logs/task-$(task_id).out
error               = logs/task-$(task_id).err
queue task_id from remaining.txt
"""

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
        serialize.save(substrate, substrates_dir / f"substrate-{label}.json.gz")

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
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "remaining.txt").write_text(
        "".join(f"{task_id}\n" for task_id in range(len(tasks)))
    )
    run_task_sh = directory / "run_task.sh"
    run_task_sh.write_text(_RUN_TASK_SH)
    run_task_sh.chmod(
        run_task_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    (directory / "pyphi.sub").write_text(
        _SUBMIT_TEMPLATE.format(
            container_image=container_image,
            request_memory=request_memory,
            request_disk=request_disk,
        )
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


def _load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _manifest_cells(manifest: dict) -> list[tuple[Any, str, tuple, tuple]]:
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
    (directory / "remaining.txt").write_text(
        "".join(f"{task_id}\n" for task_id in sorted(pending + failed))
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(manifest["tasks"]),
        n_cells=len(manifest["cells"]),
        done=tuple(done),
        failed=tuple(failed),
        pending=tuple(pending),
        total_units=float(sum(manifest["weights"])),
    )


def collect(directory: Any, partial: bool = False) -> SweepResult:
    """Reassemble the campaign's outputs into the local-sweep result.

    Cells are restored to their preparation order, so the result is
    identical to what :func:`pyphi.sweep.sweep` returns over the same axes.
    With missing or failed tasks the default is to raise with a per-task
    summary; ``partial=True`` instead warns and returns the result built
    from the completed tasks.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
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
    df = _build_df(keys, rows, cells)
    return SweepResult(df=df, results=raw, skipped=skipped)
