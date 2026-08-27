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
import os
import time
import traceback as _traceback
from pathlib import Path
from typing import Any

from pyphi import serialize
from pyphi.cache import cache_utils
from pyphi.campaign import CampaignTaskOutput
from pyphi.campaign import CellOutput
from pyphi.campaign import _parse_memory
from pyphi.campaign import _resolve_compute_ref
from pyphi.conf import config
from pyphi.cost import shard_cache_budget_bytes
from pyphi.sweep import _formalism_preset
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
            # None = the preparing session's configuration, already carried
            # by config_overrides; an explicit name applies its preset.
            **_formalism_preset(formalism),
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
        # None = the preparing session's configuration, already carried by
        # config_overrides; an explicit name applies its preset.
        **_formalism_preset(task.formalism),
        "parallel": False,
        "progress_bars": False,
    }
    # Bound the shard's caches by the memory it is actually allowed: entries
    # accumulate across every mechanism the shard carries, and the default
    # percentage of physical memory is no bound at all on a machine larger than
    # the allocation. Precedence runs from the most authoritative source down:
    # the cgroup the process runs under, then the request the scheduler was
    # asked for, then the figure planning predicted. So raising a job's request
    # raises the ceiling with it even on a pool that does not enforce memory
    # through cgroups. An explicit ceiling configured at preparation time is
    # left alone.
    spec = getattr(task, "spec", None)
    allowance = (
        cache_utils._cgroup_memory_limit()
        or _granted_memory_bytes()
        or (spec.memory_bytes if spec is not None else 0)
    )
    if allowance and overrides.get("memory_ceiling_bytes") is None:
        overrides["memory_ceiling_bytes"] = shard_cache_budget_bytes(allowance)
    return overrides


def _granted_memory_bytes() -> int:
    """The allocation the scheduler granted this job, or 0 if it said nothing.

    The submit file exports its ``request_memory`` as ``PYPHI_SHARD_MEMORY``.
    This is what the scheduler was asked for rather than what the kernel
    enforces, so it stands behind
    :func:`~pyphi.cache.cache_utils._cgroup_memory_limit` and ahead of the
    figure planning recorded — it covers a pool that grants memory without
    confining the job to it, where no cgroup limit is readable and the planned
    figure would cap the caches at the original request however much memory
    the job was actually given.
    """
    value = os.environ.get("PYPHI_SHARD_MEMORY", "").strip()
    if not value:
        return 0
    try:
        return _parse_memory(value)
    except ValueError:
        return 0


def _cache_totals() -> tuple[int, int, int]:
    """Summed (hits, misses, evictions) over this process's caches."""
    from pyphi.cache import registry

    infos = registry.info().values()
    return (
        sum(i.hits for i in infos),
        sum(i.misses for i in infos),
        sum(i.evictions for i in infos),
    )


def _spec_metrics(task: Any) -> dict[str, Any]:
    """The planned cost a shard task was packed against, if it is one."""
    spec = getattr(task, "spec", None)
    if spec is None:
        stride = getattr(task, "stride", None)
        return {"stride": list(stride)} if stride is not None else {}
    return {
        "payload_kind": spec.payload_kind,
        "units": spec.units,
        "memory_bytes": spec.memory_bytes,
        "n_mechanisms": len(spec.mechanisms) if spec.mechanisms else 1,
    }


def _global_tie_indices(ties: Any, slice_parts: list, indices: list[int]) -> list[int]:
    """Map a tie set's partitions back to global enumeration indices."""
    local = {str(p): g for p, g in zip(slice_parts, indices, strict=True)}
    return [local[str(t.partition)] for t in ties]


def _mechanism_state_pins(
    system: Any, direction: Any, mechanism: Any, purview: Any
) -> tuple:
    """Specified-state pins of the mechanism MIP search under the active
    formalism; empty for formalisms whose MIP is a plain minimum."""
    from pyphi.formalism.base import FORMALISM_REGISTRY

    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]  # pyright: ignore[reportAttributeAccessIssue]
    if not getattr(formalism, "has_state_pins", False):
        return ()
    from pyphi.formalism.iit4 import mechanism_state_pins

    return tuple(mechanism_state_pins(system, direction, mechanism, purview))


def partition_stride_entries(
    system: Any,
    direction: Any,
    mechanism: Any,
    purview: Any,
    parts: list,
    indices: list[int],
    scheme: str,
) -> list[CellOutput]:
    """Build one partition-stride cell's payloads, one entry per pin.

    φ per specified-state pin is a minimum over partitions; pin selection
    is a maximum over pins. The stride must therefore report every pin's
    local minimum — not only the pins that win locally — so the merge can
    take the cross-stride minimum per pin before selecting. The pin
    enumeration is partition-independent, so every stride reports the
    same pin set. Pin-less formalisms (e.g. IIT 3.0) report the single
    plain minimum over the stride's partitions.
    """
    from pyphi.campaign import merge as _merge
    from pyphi.formalism.queries import find_mip

    pins = _mechanism_state_pins(system, direction, mechanism, purview)
    per_pin_rias = [
        find_mip(system, direction, mechanism, purview, partitions=parts, state=pin)
        for pin in pins
    ] or [find_mip(system, direction, mechanism, purview, partitions=parts)]
    entries = []
    for pin_ria in per_pin_rias:
        pin_ties = getattr(pin_ria, "_partition_ties", None) or (pin_ria,)
        entries.append(
            CellOutput(
                status="ok",
                result=pin_ria,
                traceback=None,
                aux={
                    "pin_key": _merge._pin_key(pin_ria),
                    "pin_winner_index": _global_tie_indices((pin_ria,), parts, indices)[
                        0
                    ],
                    "tie_indices": _global_tie_indices(pin_ties, parts, indices),
                    "scheme": scheme,
                },
            )
        )
    return entries


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
                entries.extend(
                    partition_stride_entries(
                        system,
                        direction,
                        spec.mechanism,
                        spec.purview,
                        parts,
                        indices,
                        scheme,
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


def sia_stride_entries(
    system: Any, parts: list, indices: list[int], scheme: str
) -> list[CellOutput]:
    """Build one SIA stride's cell payloads.

    φ_s per (cause, effect) specified-state pair is a minimum over
    partitions; pair selection is a cascade over pairs. When pairs tie,
    the stride must report every pair's local minimum — not the winner of
    a stride-local cascade — so the merge can take the cross-stride
    minimum per pair before running the cascade globally. Per-pair
    minima are **uncapped** (MIP selection compares uncapped normalized
    φ); the merge applies the intrinsic-information cap once the global
    MIP per pair is chosen. Pin-less formalisms report the single sweep
    result.
    """
    from pyphi.formalism.base import FORMALISM_REGISTRY

    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]  # pyright: ignore[reportAttributeAccessIssue]
    if not getattr(formalism, "has_state_pins", False):
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
        return [CellOutput(status="ok", result=sia, traceback=None, aux=aux)]
    from pyphi.formalism.iit4 import sia_stride_search

    kind, payload = sia_stride_search(system, parts)
    if kind == "short_circuit":
        return [
            CellOutput(
                status="ok",
                result=payload,
                traceback=None,
                aux={"short_circuit": True, "scheme": scheme},
            )
        ]
    entries: list[CellOutput] = []
    for key, pair_sia in payload:
        ties = getattr(pair_sia, "ties", None) or (pair_sia,)
        entries.append(
            CellOutput(
                status="ok",
                result=pair_sia,
                traceback=None,
                aux={
                    "pair_key": list(key),
                    "tie_indices": _global_tie_indices(ties, parts, indices),
                    "scheme": scheme,
                },
            )
        )
    return entries


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
            return sia_stride_entries(system, parts, indices, scheme), False
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
    wall, cpu = time.perf_counter(), time.process_time()
    before = _cache_totals()
    if kind == "ces_shard":
        entries, failed = _run_ces_shard(task, substrates)
    elif kind == "sia_shard":
        entries, failed = _run_sia_shard(task, substrates)
    else:
        entries, failed = _run_sweep_task(task, substrates)
    after = _cache_totals()
    output = CampaignTaskOutput(
        task_id=task.task_id,
        pyphi_version=importlib.metadata.version("pyphi"),
        entries=tuple(entries),
        metrics={
            "wall_s": time.perf_counter() - wall,
            "cpu_s": time.process_time() - cpu,
            "cache_hits": after[0] - before[0],
            "cache_misses": after[1] - before[1],
            "cache_evictions": after[2] - before[2],
            **_spec_metrics(task),
        },
    )
    _write_output(output, Path(outputs_dir))
    return 1 if failed else 0
