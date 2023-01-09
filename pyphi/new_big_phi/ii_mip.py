# new_big_phi/ii_mip.py

from typing import Iterable, Optional

from .. import compute, utils
from ..compute.parallel import MapReduce
from ..conf import config, fallback
from ..direction import Direction
from ..models.cuts import Cut
from ..new_big_phi import (
    DEFAULT_PARTITION_CHUNKSIZE,
    DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    NullSystemIrreducibilityAnalysis,
    ShortCircuitConditions,
    SystemIrreducibilityAnalysis,
    SystemStateSpecification,
    system_intrinsic_information,
)
from ..partition import system_partitions
from ..subsystem import Subsystem


class SystemIrreducibilityAnalysisII(SystemIrreducibilityAnalysis):
    def __init__(
        self,
        *args,
        partitioned_system_state: Optional[SystemStateSpecification] = None,
        **kwargs,
    ):
        self.partitioned_system_state = partitioned_system_state
        super().__init__(*args, **kwargs)

    def _repr_columns(self):
        return super()._repr_columns() + self.partitioned_system_state._repr_columns(
            prefix="Partitioned "
        )


def normalization_factor(partition: Cut) -> float:
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemStateSpecification,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
) -> SystemIrreducibilityAnalysisII:
    repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
    directions = fallback(directions, Direction.both())

    cut_subsystem = subsystem.apply_cut(partition)
    partitioned_system_state = system_intrinsic_information(
        cut_subsystem,
        repertoire_distance=repertoire_distance,
        system_state=system_state,
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


def sia(
    subsystem: Subsystem,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
    partitions=None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    # TODO: trivial reducibility

    partitions = system_partitions(
        subsystem.node_indices,
        node_labels=subsystem.node_labels,
        partition_scheme=partitions,
    )

    system_state = system_intrinsic_information(
        subsystem, repertoire_distance=repertoire_distance
    )

    default_sia = NullSystemIrreducibilityAnalysis(
        reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS],
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )

    kwargs = {
        "parallel": config.PARALLEL_CUT_EVALUATION,
        "progress": config.PROGRESS_BARS,
        "chunksize": DEFAULT_PARTITION_CHUNKSIZE,
        "sequential_threshold": DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
        **kwargs,
    }
    return MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs=dict(
            subsystem=subsystem,
            system_state=system_state,
            repertoire_distance=repertoire_distance,
            directions=directions,
        ),
        reduce_func=min,
        reduce_kwargs=dict(default=default_sia),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **kwargs,
    ).run()
