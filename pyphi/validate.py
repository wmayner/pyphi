# validate.py
"""Methods for validating user input."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np

from . import exceptions
from .conf import config
from .direction import Direction

# pylint: disable=redefined-outer-name


# TODO(4.0) move to `Direction`
def directions(directions: Iterable[Direction], **kwargs: bool) -> bool:
    """Validate each direction in an iterable.

    Args:
        directions (Iterable[Direction]): Directions to validate.
        **kwargs: Passed through to |direction|.

    Returns:
        bool: ``True`` if every element is a valid |Direction|.
    """
    return all(direction(d, **kwargs) for d in directions)


def direction(direction: Direction, allow_bi: bool = False) -> bool:
    """Validate that the given direction is one of the allowed constants.

    Args:
        direction (Direction): Direction to validate.
        allow_bi (bool): Whether bidirectional arrows are allowed.

    Returns:
        bool: ``True`` if the direction is valid; otherwise raises.
    """
    valid = set(Direction.both())
    if allow_bi:
        valid.add(Direction.BIDIRECTIONAL)

    if direction not in valid:
        raise ValueError(
            f"`direction` must be one of `Direction.{valid}`; "
            f"got {type(direction)} `{direction}`"
        )

    return True


def connectivity_matrix(cm: np.ndarray) -> bool:
    """Validate the given connectivity matrix."""
    # Special case for empty matrices.
    if cm.size == 0:
        return True
    if cm.ndim != 2:
        raise ValueError("Connectivity matrix must be 2-dimensional.")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if not np.all(np.logical_or(cm == 1, cm == 0)):
        raise ValueError("Connectivity matrix must contain only binary values.")
    return True


def node_labels(node_labels: Sequence[str], node_indices: Sequence[int]) -> None:
    """Validate that there is a label for each node."""
    if len(node_labels) != len(node_indices):
        raise ValueError(f"Labels {node_labels} must label every node {node_indices}.")

    if len(node_labels) != len(set(node_labels)):
        raise ValueError(f"Labels {node_labels} must be unique.")


def substrate(n: object) -> bool:
    """Validate a |Substrate|.

    Checks the TPM and connectivity matrix.
    """
    n.tpm.validate()  # type: ignore[attr-defined]
    connectivity_matrix(n.cm)  # type: ignore[attr-defined]
    if n.cm.shape[0] != n.size:  # type: ignore[attr-defined]
        raise ValueError(
            "Connectivity matrix must be NxN, where N is the "
            "number of nodes in the substrate."
        )
    return True


def is_substrate(substrate: object) -> None:
    """Validate that the argument is a |Substrate|."""
    from . import Substrate

    if not isinstance(substrate, Substrate):
        raise ValueError(
            "Input must be a Substrate (perhaps you passed a System instead?"
        )


def node_states(state: Sequence[int]) -> None:
    """Check that the state contains only zeros and ones."""
    if not all(n in (0, 1) for n in state):
        raise ValueError("Invalid state: states must consist of only zeros and ones.")


def state_length(state: Sequence[int], size: int) -> bool:
    """Check that the state is the given size."""
    if len(state) != size:
        raise ValueError(
            "Invalid state: there must be one entry per "
            f"node in the substrate; this state has {len(state)} entries, but "
            f"there are {size} nodes."
        )
    return True


def state_reachable(system: object) -> None:
    """Return whether a state can be reached according to the substrate's TPM."""
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    # First we take the submatrix of the conditioned TPM that corresponds to
    # the nodes that are actually in the system...
    tpm = system.effect_tpm.tpm[..., system.node_indices]  # type: ignore[attr-defined]
    # Then we do the subtraction and test.
    test = tpm - np.array(system.proper_state)  # type: ignore[attr-defined]
    if not np.any(np.logical_and(test > -1, test < 1).all(-1)):
        raise exceptions.StateUnreachableForwardsError(system.state)  # type: ignore[attr-defined]


def system_partition(partition: object, node_indices: Sequence[int]) -> None:
    """Check that the partition covers only the given nodes."""
    if set(partition.indices) != set(node_indices):  # type: ignore[attr-defined]
        raise ValueError(
            f"{partition} nodes are not equal to system nodes {node_indices}"
        )


def system(s: object) -> bool:
    """Validate a |System|.

    Checks its state and partition.
    """
    node_states(s.state)  # type: ignore[attr-defined]
    system_partition(s.partition, s.partition_indices)  # type: ignore[attr-defined]
    if config.infrastructure.validate_system_states:
        state_reachable(s)
    return True


def time_scale(time_scale: int) -> None:
    """Validate a macro temporal time scale."""
    if time_scale <= 0 or isinstance(time_scale, float):
        raise ValueError("time scale must be a positive integer")


def partition(partition: Iterable[Iterable[int]]) -> None:
    """Validate a partition - used by blackboxes and coarse grains."""
    nodes = set()
    for part in partition:
        for node in part:
            if node in nodes:
                raise ValueError(
                    f"Micro-element {node} may not be partitioned into multiple "
                    "macro-elements"
                )
            nodes.add(node)


def coarse_grain(coarse_grain: object) -> None:
    """Validate a macro coarse-graining."""
    partition(coarse_grain.partition)  # type: ignore[attr-defined]

    if len(coarse_grain.partition) != len(coarse_grain.grouping):  # type: ignore[attr-defined]
        raise ValueError("output and state groupings must be the same size")

    for part, group in zip(coarse_grain.partition, coarse_grain.grouping, strict=False):  # type: ignore[attr-defined]
        if set(range(len(part) + 1)) != set(group[0] + group[1]):
            # Check that all elements in the partition are in one of the two
            # state groupings
            raise ValueError(
                f"elements in output grouping {part} do not match "
                f"elements in state grouping {group}"
            )


def blackbox(blackbox: object) -> None:
    """Validate a macro blackboxing."""
    if tuple(sorted(blackbox.output_indices)) != blackbox.output_indices:  # type: ignore[attr-defined]
        raise ValueError(f"Output indices {blackbox.output_indices} must be ordered")  # type: ignore[attr-defined]

    partition(blackbox.partition)  # type: ignore[attr-defined]

    for part in blackbox.partition:  # type: ignore[attr-defined]
        if not set(part) & set(blackbox.output_indices):  # type: ignore[attr-defined]
            raise ValueError(f"Every blackbox must have an output - {part} does not")


def blackbox_and_coarse_grain(
    blackbox: object | None, coarse_grain: object | None
) -> None:
    """Validate that a coarse-graining properly combines the outputs of a
    blackboxing.
    """
    if blackbox is None:
        return

    for box in blackbox.partition:  # type: ignore[attr-defined]
        # Outputs of the box
        outputs = set(box) & set(blackbox.output_indices)  # type: ignore[attr-defined]

        if coarse_grain is None and len(outputs) > 1:
            raise ValueError(
                "A blackboxing with multiple outputs per box must be coarse-grained."
            )

        if coarse_grain and not any(
            outputs.issubset(part)
            for part in coarse_grain.partition  # type: ignore[attr-defined]
        ):
            raise ValueError(
                "Multiple outputs from a blackbox must be partitioned into "
                "the same macro-element of the coarse-graining"
            )


def relata(relata: Iterable[object] | None) -> None:
    """Validate a set of relata."""
    if not relata:
        raise ValueError("relata cannot be empty")
