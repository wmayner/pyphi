"""Execute one campaign task file and write its output document.

The runner is a fixed entry point (``python -m pyphi.campaign run``) that
behaves identically inside the campaign's container, in a local shell, and
under test: it loads the task, loads the substrates it references, installs
the shipped configuration beneath the task's formalism preset, runs the
task's items in order, and atomically writes one output document holding a
per-item outcome. The process exit code is nonzero when any item errored,
so scheduler logs reflect failures, but the output document is written in
every case.
"""

from __future__ import annotations

import importlib.metadata
import traceback as _traceback
from pathlib import Path
from typing import Any

from pyphi import serialize
from pyphi.cache import cache_utils
from pyphi.campaign import CampaignTaskOutput
from pyphi.campaign import CellOutput
from pyphi.campaign import _resolve_compute_ref
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.cost import shard_cache_budget_bytes
from pyphi.sweep import _run_cell
from pyphi.sweep import _Skipped

__all__ = ["run_task"]


def _task_labels(task: Any) -> set:
    if hasattr(task, "cells"):
        return {cell[0] for cell in task.cells}
    return {task.substrate_label}


def _load_substrates(task: Any, substrates_dir: Path) -> dict:
    return {
        label: serialize.load(substrates_dir / f"substrate-{label}.msgpack.gz")
        for label in _task_labels(task)
    }


def _write_output(output: CampaignTaskOutput, outputs_dir: Path) -> None:
    final = outputs_dir / f"task-{output.task_id:04d}.json.gz"
    if final.exists():
        n = 1
        while (
            attempt := final.with_name(f"task-{output.task_id:04d}.attempt-{n}.json.gz")
        ).exists():
            n += 1
        final.rename(attempt)
    # The temporary name keeps the .json.gz suffixes so format inference is
    # unchanged; Path.replace makes the final path appear atomically.
    tmp = final.with_name(f".tmp-{final.name}")
    serialize.save(output, tmp)
    tmp.replace(final)


def _run_sweep_task(task: Any, substrates: dict) -> tuple[list[CellOutput], bool]:
    compute = (
        task.compute
        if task.compute is not None
        else _resolve_compute_ref(task.compute_ref)
    )
    entries: list[CellOutput] = []
    failed = False
    for label, formalism, subset, state in task.cells:
        overrides = {
            **task.config_overrides,
            **presets.by_name[formalism],
            "parallel": False,
            "progress_bars": False,
        }
        try:
            with config.override(**overrides):
                result = _run_cell(
                    (label, subset, state),
                    substrates=substrates,
                    compute=compute,
                    skip=task.skip_uncomputable,
                )
            if isinstance(result, _Skipped):
                entries.append(CellOutput(status="skipped", result=None, traceback=None))
            else:
                entries.append(CellOutput(status="ok", result=result, traceback=None))
        except Exception:
            entries.append(
                CellOutput(
                    status="error", result=None, traceback=_traceback.format_exc()
                )
            )
            failed = True
    return entries, failed


def _shard_config(task: Any) -> dict[str, Any]:
    overrides = {
        **task.config_overrides,
        **presets.by_name[task.formalism],
        "parallel": False,
        "progress_bars": False,
    }
    # Bound the shard's caches by the memory it is actually allowed: entries
    # accumulate across every mechanism the shard carries, and the default
    # percentage of physical memory is no bound at all on a machine larger than
    # the allocation. The allocation the process runs under takes precedence
    # over the one planning predicted, so raising a job's request raises the
    # ceiling with it; the planned figure stands in where the allocation cannot
    # be read. An explicit ceiling configured at preparation time is left alone.
    spec = getattr(task, "spec", None)
    if (
        spec is not None
        and spec.memory_bytes
        and overrides.get("maximum_cache_memory_bytes") is None
    ):
        allowance = cache_utils._cgroup_memory_limit() or spec.memory_bytes
        overrides["maximum_cache_memory_bytes"] = shard_cache_budget_bytes(allowance)
    return overrides


def _global_tie_indices(ties: Any, slice_parts: list, indices: list[int]) -> list[int]:
    """Map a tie set's partitions back to global enumeration indices."""
    local = {str(p): g for p, g in zip(slice_parts, indices, strict=True)}
    return [local[str(t.partition)] for t in ties]


def _run_ces_shard(task: Any, substrates: dict) -> tuple[list[CellOutput], bool]:
    from pyphi.campaign import shards as _shards
    from pyphi.direction import Direction
    from pyphi.formalism.queries import distinction as _distinction
    from pyphi.formalism.queries import find_mip
    from pyphi.system import System

    entries: list[CellOutput] = []
    failed = False
    spec = task.spec
    with config.override(**_shard_config(task)):
        system = System(
            substrates[task.substrate_label], task.state, node_indices=task.subset
        )
        scheme = config.formalism.iit.mechanism_partition_scheme  # pyright: ignore[reportAttributeAccessIssue]
        try:
            if spec.payload_kind == "mechanisms":
                for mechanism in spec.mechanisms:
                    cause_axis = task.scope.purview_axis(Direction.CAUSE, mechanism)
                    cause_purviews = list(
                        cause_axis.select(
                            system.potential_purviews(
                                Direction.CAUSE,
                                mechanism,
                                max_order=cause_axis.order_bound(),
                            )
                        )
                    )
                    effect_axis = task.scope.purview_axis(Direction.EFFECT, mechanism)
                    effect_purviews = list(
                        effect_axis.select(
                            system.potential_purviews(
                                Direction.EFFECT,
                                mechanism,
                                max_order=effect_axis.order_bound(),
                            )
                        )
                    )
                    result = _distinction(
                        system,
                        mechanism,
                        cause_purviews=cause_purviews,
                        effect_purviews=effect_purviews,
                    )
                    entries.append(
                        CellOutput(status="ok", result=result, traceback=None)
                    )
            elif spec.payload_kind == "purview_range":
                direction = Direction[spec.direction]
                for purview in spec.purviews:
                    ria = find_mip(system, direction, spec.mechanism, purview)
                    entries.append(CellOutput(status="ok", result=ria, traceback=None))
            elif spec.payload_kind == "partition_stride":
                direction = Direction[spec.direction]
                i, k = spec.stride
                parts, indices = _shards.enumerate_partition_stride(
                    spec.mechanism, spec.purview, system.node_labels, i, k
                )
                if task.ordering == "bottleneck_first":
                    parts, indices = _shards.bottleneck_order(
                        parts, indices, system.cm, direction
                    )
                ria = find_mip(
                    system,
                    direction,
                    spec.mechanism,
                    spec.purview,
                    partitions=parts,
                )
                tie_indices = {}
                pin_winner_indices = {}
                for pin in getattr(ria, "_state_ties", None) or (ria,):
                    key = repr(pin.specified_state.state)
                    pin_ties = getattr(pin, "_partition_ties", None) or (pin,)
                    tie_indices[key] = _global_tie_indices(pin_ties, parts, indices)
                    pin_winner_indices[key] = _global_tie_indices(
                        (pin,), parts, indices
                    )[0]
                entries.append(
                    CellOutput(
                        status="ok",
                        result=ria,
                        traceback=None,
                        aux={
                            "tie_indices": tie_indices,
                            "pin_winner_indices": pin_winner_indices,
                            "scheme": scheme,
                        },
                    )
                )
            else:
                raise ValueError(f"unknown payload kind {spec.payload_kind!r}")
        except Exception:
            entries.append(
                CellOutput(
                    status="error", result=None, traceback=_traceback.format_exc()
                )
            )
            failed = True
    return entries, failed


def _run_sia_shard(task: Any, substrates: dict) -> tuple[list[CellOutput], bool]:
    from pyphi.campaign import shards as _shards
    from pyphi.system import System

    with config.override(**_shard_config(task)):
        system = System(
            substrates[task.substrate_label], task.state, node_indices=task.subset
        )
        scheme = config.formalism.iit.system_partition_scheme  # pyright: ignore[reportAttributeAccessIssue]
        i, k = task.stride
        parts, indices = _shards.enumerate_system_partition_stride(system, scheme, i, k)
        try:
            sia = system.sia(partitions=parts)
            if getattr(sia, "reasons", None):
                # A null short-circuit (e.g. no strong connectivity) never
                # consults the partition restriction, so every stride of the
                # cell produces this same result.
                aux = {"short_circuit": True, "scheme": scheme}
            else:
                ties = getattr(sia, "ties", None) or (sia,)
                aux = {
                    "tie_indices": _global_tie_indices(ties, parts, indices),
                    "scheme": scheme,
                }
            return (
                [CellOutput(status="ok", result=sia, traceback=None, aux=aux)],
                False,
            )
        except Exception:
            return (
                [
                    CellOutput(
                        status="error",
                        result=None,
                        traceback=_traceback.format_exc(),
                    )
                ],
                True,
            )


def run_task(
    task_path: Any,
    substrates_dir: Any = "substrates",
    outputs_dir: Any = ".",
) -> int:
    """Run one task file; return 0 if every item is ok or skipped, else 1.

    Parameters
    ----------
    task_path
        Path to a serialized campaign task of any kind.
    substrates_dir
        Directory holding the campaign's serialized substrates.
    outputs_dir
        Directory to write ``task-<id>.json.gz`` into (atomically; a
        pre-existing output is preserved under an ``attempt-<n>`` name).
    """
    task = serialize.load(task_path)
    substrates = _load_substrates(task, Path(substrates_dir))
    kind = getattr(task, "kind", "sweep_cells")
    if kind == "ces_shard":
        entries, failed = _run_ces_shard(task, substrates)
    elif kind == "sia_shard":
        entries, failed = _run_sia_shard(task, substrates)
    else:
        entries, failed = _run_sweep_task(task, substrates)
    output = CampaignTaskOutput(
        task_id=task.task_id,
        pyphi_version=importlib.metadata.version("pyphi"),
        entries=tuple(entries),
    )
    _write_output(output, Path(outputs_dir))
    return 1 if failed else 0
