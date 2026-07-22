"""Shard planning for scoped cause-effect campaigns.

The planner descends a three-rung ladder, splitting only where the per-job
budget requires: whole mechanisms are cost-balance-packed into shards; a
mechanism over budget splits its scoped (direction, purview) list into
cost-balanced ranges; a single (mechanism, direction, purview) pair over
budget splits its partition enumeration into interleaved strides (shard i
of k evaluates partitions i, i+k, i+2k, …), which balances any systematic
cost trend along the enumeration. Sharding never changes results — every
shard executes exact computations over a subset, and collection merges tie
sets losslessly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import islice
from typing import Any

from pyphi.conf import config
from pyphi.cost import MechanismWorkload
from pyphi.cost import mechanism_workloads
from pyphi.cost import partition_sweep_count
from pyphi.cost import round_memory_bytes
from pyphi.cost import shard_memory_bytes
from pyphi.direction import Direction
from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partition_types
from pyphi.warnings import PyPhiWarning

__all__ = [
    "ShardSpec",
    "bottleneck_order",
    "cut_present_edges",
    "enumerate_partition_stride",
    "enumerate_system_partition_stride",
    "plan_ces_shards",
    "plan_sia_shards",
]


@dataclass(frozen=True)
class ShardSpec:
    """One shard of a scoped analysis: what to compute and how it was split."""

    payload_kind: str
    mechanisms: tuple[tuple[int, ...], ...] = ()
    mechanism: tuple[int, ...] | None = None
    direction: str | None = None
    purviews: tuple[tuple[int, ...], ...] = ()
    purview: tuple[int, ...] | None = None
    stride: tuple[int, int] | None = None
    units: float = 0.0
    memory_bytes: int = 0


def enumerate_partition_stride(
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    node_labels: Any,
    i: int,
    k: int,
) -> tuple[list, list[int]]:
    """Materialize stride ``i`` of ``k`` of the partition enumeration.

    Returns the partitions and their global enumeration indices. Only the
    stride is materialized; the full enumeration is consumed lazily. The
    enumeration must use the same ``node_labels`` the analysis uses, so
    partition identities (their string forms) agree across processes.
    """
    parts = list(
        islice(mechanism_partitions(mechanism, purview, node_labels), i, None, k)
    )
    return parts, [i + j * k for j in range(len(parts))]


def enumerate_system_partition_stride(
    system: Any, scheme: str, i: int, k: int
) -> tuple[list, list[int]]:
    """Materialize stride ``i`` of ``k`` of the system-partition enumeration."""
    generator = system_partition_types[scheme](
        system.partition_indices, node_labels=system.node_labels
    )
    parts = list(islice(generator, i, None, k))
    return parts, [i + j * k for j in range(len(parts))]


def cut_present_edges(partition: Any, cm: Any, direction: Direction) -> int:
    """Count present-in-cm connections severed by a mechanism partition."""
    parts = list(partition)
    count = 0
    for a, part_a in enumerate(parts):
        for b, part_b in enumerate(parts):
            if a == b:
                continue
            for m in part_a.mechanism:
                for p in part_b.purview:
                    src, dst = (m, p) if direction == Direction.EFFECT else (p, m)
                    if cm[src, dst]:
                        count += 1
    return count


def bottleneck_order(
    partitions: list, indices: list[int], cm: Any, direction: Direction
) -> tuple[list, list[int]]:
    """Reorder a partition slice so likely-reducible partitions come first.

    Sorts by ascending count of severed present connections: a partition
    that cuts no present connection yields φ = 0, so on sparse substrates
    the sweep's zero-φ short-circuit fires within the first evaluations.
    Ordering never affects results — the minimum is order-independent and
    tie resolution runs on the collected set — only time to short-circuit.
    The sort is stable, so equal-count partitions keep enumeration order.
    """
    keyed = sorted(
        zip(partitions, indices, strict=True),
        key=lambda pair: cut_present_edges(pair[0], cm, direction),
    )
    return [p for p, _ in keyed], [i for _, i in keyed]


def _memory_class(cells: int, floor: int) -> int:
    """The rounded, floored memory request for a shard holding ``cells``."""
    return max(floor, round_memory_bytes(shard_memory_bytes(cells)))


def _pack_specs(items: list[ShardSpec], units_per_job: float) -> list[ShardSpec]:
    """Cost-balance whole-mechanism items into "mechanisms" shards.

    Packing runs within each memory class, so one large-purview mechanism
    never inflates the request of a shard of small ones.
    """
    packed: list[ShardSpec] = []
    for memory in sorted({s.memory_bytes for s in items}):
        group = [s for s in items if s.memory_bytes == memory]
        weights = [s.units for s in group]
        jobs = max(1, math.ceil(sum(weights) / units_per_job))
        bins = cost_balanced_partition(weights, jobs)
        packed.extend(
            ShardSpec(
                payload_kind="mechanisms",
                mechanisms=tuple(m for i in indices for m in group[i].mechanisms),
                units=float(sum(group[i].units for i in indices)),
                memory_bytes=memory,
            )
            for indices in (sorted(b) for b in bins)
        )
    return packed


def plan_ces_shards(
    system: Any,
    scope: Any,
    units_per_job: float,
    limit: int = 10_000_000,
    workloads: dict[tuple[int, ...], MechanismWorkload] | None = None,
    memory_floor_bytes: int = 0,
) -> list[ShardSpec]:
    """Plan the shards of a scoped cause-effect computation.

    Descends mechanism → purview-range → partition-stride only where the
    budget requires. Deterministic for fixed inputs; every spec carries its
    estimated work units and its rounded memory request.

    Parameters
    ----------
    system
        The system to analyze.
    scope
        The resolved feasibility surface.
    units_per_job : float
        Target work units per shard.
    limit : int, optional
        Work budget for the counting walk (ignored when ``workloads`` is
        given).
    workloads : dict, optional
        A precomputed :func:`pyphi.cost.mechanism_workloads` mapping for
        the same system and scope; when given, the walk is not repeated.
    memory_floor_bytes : int, optional
        Minimum per-shard memory request; every shard's ``memory_bytes``
        is at least this.
    """
    if workloads is None:
        workloads = mechanism_workloads(
            system.substrate, subset=system.node_indices, scope=scope, limit=limit
        )
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    whole: list[ShardSpec] = []
    specs: list[ShardSpec] = []
    for mechanism, workload in workloads.items():
        units = workload.units
        if units <= units_per_job:
            whole.append(
                ShardSpec(
                    payload_kind="mechanisms",
                    mechanisms=(mechanism,),
                    units=float(units),
                    memory_bytes=_memory_class(
                        workload.max_repertoire_cells, memory_floor_bytes
                    ),
                )
            )
            continue
        # Rung 2: split this mechanism's (direction, purview) list.
        for direction in (Direction.CAUSE, Direction.EFFECT):
            purviews = system.potential_purviews(direction, mechanism)
            purviews = list(scope.purview_axis(direction, mechanism).select(purviews))
            if not purviews:
                continue
            triples = [
                (
                    p,
                    1.0 + partition_sweep_count(len(mechanism), len(p)),
                    _memory_class(math.prod(alphabet[u] for u in p), memory_floor_bytes),
                )
                for p in purviews
            ]
            oversized = [(p, w, m) for p, w, m in triples if w > units_per_job]
            fitting = [(p, w, m) for p, w, m in triples if w <= units_per_job]
            # Pack fitting purviews within each memory class, so one large
            # purview never inflates the request of a shard of small ones.
            for memory in sorted({m for _, _, m in fitting}):
                group = [(p, w) for p, w, m in fitting if m == memory]
                jobs = max(1, math.ceil(sum(w for _, w in group) / units_per_job))
                bins = cost_balanced_partition([w for _, w in group], jobs)
                specs.extend(
                    ShardSpec(
                        payload_kind="purview_range",
                        mechanism=mechanism,
                        direction=direction.name,
                        purviews=tuple(group[i][0] for i in bin_indices),
                        units=float(sum(group[i][1] for i in bin_indices)),
                        memory_bytes=memory,
                    )
                    for bin_indices in (sorted(b) for b in bins)
                )
            # Rung 3: stride each oversized pair.
            for purview, weight, memory in oversized:
                count = partition_sweep_count(len(mechanism), len(purview))
                k = min(math.ceil(weight / units_per_job), count)
                if weight / k > units_per_job:
                    warnings.warn(
                        f"budget units_per_job={units_per_job:.3g} is "
                        f"unreachable for mechanism {mechanism} purview "
                        f"{purview} ({count} partitions); one partition per "
                        "shard is the floor",
                        PyPhiWarning,
                        stacklevel=2,
                    )
                specs.extend(
                    ShardSpec(
                        payload_kind="partition_stride",
                        mechanism=mechanism,
                        direction=direction.name,
                        purview=purview,
                        stride=(i, k),
                        units=float(weight / k),
                        memory_bytes=memory,
                    )
                    for i in range(k)
                )
    return _pack_specs(whole, units_per_job) + specs


def plan_sia_shards(
    system: Any, units_per_job: float, memory_floor_bytes: int = 0
) -> list[ShardSpec]:
    """Plan system-partition strides for the system irreducibility analysis."""
    scheme = config.formalism.iit.system_partition_scheme
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    cells = math.prod(alphabet[u] for u in system.node_indices)
    memory = _memory_class(cells, memory_floor_bytes)
    total = sum(
        1
        for _ in system_partition_types[scheme](
            system.partition_indices, node_labels=system.node_labels
        )
    )
    k = max(1, min(math.ceil(total / units_per_job), total))
    return [
        ShardSpec(
            payload_kind="partition_stride",
            mechanism=None,
            stride=(i, k),
            units=float(total / k),
            memory_bytes=memory,
        )
        for i in range(k)
    ]
