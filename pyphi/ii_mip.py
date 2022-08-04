# ii_mip.py

from typing import Optional, Iterable

from . import compute
from .conf import config, fallback
from .direction import Direction
from .models.cuts import Cut
from .new_big_phi import (
    SystemIrreducibilityAnalysis,
    SystemState,
    find_system_state,
)
from .partition import system_directed_bipartitions
from .subsystem import Subsystem

DEFAULT_SEQUENTIAL_THRESHOLD = 2 ** 4
DEFAULT_CHUNKSIZE = 4 * DEFAULT_SEQUENTIAL_THRESHOLD


class SystemIrreducibilityAnalysisII(SystemIrreducibilityAnalysis):
    def __init__(
        self, *args, partitioned_system_state: Optional[SystemState] = None, **kwargs
    ):
        self.partitioned_system_state = partitioned_system_state
        super().__init__(*args, **kwargs)

    def _repr_columns(self):
        return super()._repr_columns() + self.partitioned_system_state._repr_columns(
            prefix="Partitioned "
        )


def normalization_factor(partition: Cut) -> float:
    # TODO!!!
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
) -> SystemIrreducibilityAnalysisII:
    repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
    directions = fallback(directions, Direction.both())

    cut_subsystem = subsystem.apply_cut(partition)
    partitioned_system_state = find_system_state(
        cut_subsystem, repertoire_distance=repertoire_distance
    )

    integration = {
        direction: (
            system_state.intrinsic_information[direction]
            - partitioned_system_state.intrinsic_information[direction]
        )
        for direction in Direction.both()
    }
    phi = min(integration[direction] for direction in directions)
    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    return SystemIrreducibilityAnalysisII(
        phi=phi,
        normalized_phi=normalized_phi,
        partition=partition,
        phi_cause=integration[Direction.CAUSE],
        phi_effect=integration[Direction.EFFECT],
        system_state=system_state,
        partitioned_system_state=partitioned_system_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_SEQUENTIAL_THRESHOLD,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    # TODO: trivial reducibility

    system_state = find_system_state(subsystem, repertoire_distance=repertoire_distance)
    partitions = system_directed_bipartitions(subsystem.node_indices)

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
