# new_big_phi/unidirectional_vertical.py

from typing import Iterable, Optional, Union

from .. import compute, connectivity
from ..conf import config, fallback
from ..direction import Direction
from ..metrics.distribution import repertoire_distance as _repertoire_distance
from ..models.cuts import Cut, GeneralKCut
from ..new_big_phi import (
    DEFAULT_PARTITION_CHUNKSIZE,
    DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    SystemIrreducibilityAnalysis,
    SystemState,
    find_system_state,
)
from ..partition import system_partitions
from ..subsystem import Subsystem


def normalization_factor(partition: Union[Cut, GeneralKCut]) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def integration_value(
    direction: Direction,
    subsystem: Subsystem,
    cut_subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
) -> float:
    unpartitioned_repertoire = subsystem.repertoire(
        direction, subsystem.node_indices, subsystem.node_indices
    )
    partitioned_repertoire = cut_subsystem.repertoire(
        direction, subsystem.node_indices, subsystem.node_indices
    )
    return (
        _repertoire_distance(
            unpartitioned_repertoire,
            partitioned_repertoire,
            repertoire_distance=repertoire_distance,
            state=system_state[direction],
        ),
        unpartitioned_repertoire,
        partitioned_repertoire,
    )


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
) -> SystemIrreducibilityAnalysis:
    directions = fallback(directions, Direction.both())

    cut_subsystem = subsystem.apply_cut(partition)

    integration = {
        direction: integration_value(
            direction,
            subsystem,
            cut_subsystem,
            system_state,
            repertoire_distance=repertoire_distance,
        )
        for direction in Direction.both()
    }
    phi_c, repertoire_c, partitioned_repertoire_c = integration[Direction.CAUSE]
    phi_e, repertoire_e, partitioned_repertoire_e = integration[Direction.EFFECT]

    phi = min(integration[direction][0] for direction in directions)
    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    return SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        phi_cause=phi_c,
        phi_effect=phi_e,
        partition=partition,
        repertoire_cause=repertoire_c,
        partitioned_repertoire_cause=partitioned_repertoire_c,
        repertoire_effect=repertoire_e,
        partitioned_repertoire_effect=partitioned_repertoire_e,
        system_state=system_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
    partitions: str = None,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    partitions = fallback(partitions, config.SYSTEM_PARTITION_TYPE)
    # TODO: trivial reducibility

    filter_func = None
    if partitions == "GENERAL":

        def is_disconnecting_partition(partition):
            return not connectivity.is_strong(subsystem.apply_cut(partition).cm)

        filter_func = is_disconnecting_partition

    partitions = system_partitions(
        subsystem.node_indices,
        node_labels=subsystem.node_labels,
        partition_scheme=partitions,
        filter_func=filter_func,
    )

    system_state = find_system_state(subsystem)

    return compute.parallel.map_reduce(
        evaluate_partition,
        min,
        partitions,
        subsystem=subsystem,
        system_state=system_state,
        repertoire_distance=repertoire_distance,
        directions=directions,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )
