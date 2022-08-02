import pyphi
from dataclasses import dataclass
from pyphi import Direction
from pyphi.conf import fallback
from pyphi.models import cmp, fmt
from pyphi.models.cuts import KPartition


def normalization_factor(subsystem, partition):
    if isinstance(partition, pyphi.partition.CompletePartition):
        return len(subsystem) ** 2
    return sum(
        [
            # NOTE: Assumes mechanism and purview are the same in each part
            len(part.mechanism) * (len(subsystem) - len(part.mechanism))
            for part in partition
        ]
    ) * len(subsystem)


def system_set_partitions(collection, node_labels=None):
    collection = tuple(collection)
    set_partitions = pyphi.partition.partitions(collection)
    for partition in set_partitions:
        if len(partition) == 1:
            yield pyphi.partition.complete_partition(collection, collection)
        else:
            yield pyphi.partition.KPartition(
                *[pyphi.partition.Part(tuple(part), tuple(part)) for part in partition],
                node_labels=node_labels,
            )


def evaluate_system_partition(
    partition,
    subsystem,
    sys_effect_state,
    sys_cause_state,
    effect_only=False,
):
    effect_info, _ = subsystem.evaluate_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        partition=partition,
        state=sys_effect_state,
    )
    if effect_only:
        phi = effect_info
    else:
        cause_info, _ = subsystem.evaluate_partition(
            Direction.CAUSE,
            mechanism=subsystem.node_indices,
            purview=subsystem.node_indices,
            partition=partition,
            state=sys_cause_state,
        )
        phi = min(effect_info, cause_info)

    norm = normalization_factor(subsystem, partition)
    normalized_phi = phi / norm
    return SystemIrreducibilityAnalysis(
        phi=phi,
        norm=norm,
        normalized_phi=normalized_phi,
        cause_state=sys_cause_state,
        effect_state=sys_effect_state,
        partition=partition,
    )


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    norm: float
    normalized_phi: float
    cause_state: tuple
    effect_state: tuple
    partition: KPartition
    effect_only: bool = False

    _sia_attributes = ["phi", "cause_state", "effect_state", "partition"]

    def order_by(self):
        return self.normalized_phi

    def __eq__(self, other):
        return cmp.general_eq(self, other, self._sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has |big_phi > 0|."""
        return pyphi.utils.is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.cause_state,
                self.effect_state,
                self.partition,
            )
        )

    def __repr__(self):
        body = ""
        for line in reversed(
            [
                f" Cause state: {self.cause_state}",
                f"Effect state: {self.effect_state}",
                f" Effect only: {self.effect_only}",
            ]
        ):
            body = fmt.header(line, body)
        body = "\n" + fmt.header(str(self.partition), body, under_char=" ")
        body = (
            "\n".join(
                fmt.align_columns(
                    [
                        (fmt.BIG_PHI, self.phi),
                        ("Norm", self.norm),
                        (f"Normalized {fmt.BIG_PHI}", self.normalized_phi),
                        ("Partition", ""),
                    ]
                )
            )
            + "\n"
            + body
        )
        body = fmt.header(
            "System irreducibility analysis", body, under_char=fmt.HEADER_BAR_2
        )
        return fmt.box(fmt.center(body))


def null_sia():
    return SystemIrreducibilityAnalysis(
        phi=0.0,
        normalized_phi=0.0,
        norm=0.0,
        cause_state=None,
        effect_state=None,
        partition=None,
    )


def find_mip(subsystem, effect_only=False, parallel=None, progress=True):
    # NOTE: Tie-breaking happens here when we access the first element.
    sys_effect_state = subsystem.find_maximal_state_under_complete_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]
    sys_cause_state = subsystem.find_maximal_state_under_complete_partition(
        Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]
    return pyphi.compute.parallel.map_reduce(
        evaluate_system_partition,
        min,
        system_set_partitions(
            subsystem.node_indices, node_labels=subsystem.node_labels
        ),
        subsystem=subsystem,
        sys_cause_state=sys_cause_state,
        sys_effect_state=sys_effect_state,
        effect_only=effect_only,
        # Parallel kwargs
        chunksize=2 ** 15,
        sequential_threshold=2 ** 14,
        parallel=parallel,
        shortcircuit_value=null_sia(),
        progress=fallback(progress, pyphi.config.PROGRESS_BARS),
        desc="Partitions",
    )
