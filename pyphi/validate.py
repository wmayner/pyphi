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

    Checks the FactoredTPM and connectivity matrix.
    """
    from pyphi.core.tpm.factored import FactoredTPM

    factored = n.factored_tpm  # type: ignore[attr-defined]
    if not isinstance(factored, FactoredTPM):
        raise ValueError("substrate.factored_tpm must be a FactoredTPM")
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


def node_states(state: Sequence[int], alphabet_sizes: Sequence[int]) -> None:
    """Check that each state entry is a valid index into its node's alphabet.

    Args:
        state: Per-node state indices.
        alphabet_sizes: Per-node alphabet sizes; ``state[i]`` must satisfy
            ``0 <= state[i] < alphabet_sizes[i]``.
    """
    if len(state) != len(alphabet_sizes):
        raise ValueError(
            f"State length {len(state)} does not match alphabet_sizes length "
            f"{len(alphabet_sizes)}."
        )
    for i, (s, k) in enumerate(zip(state, alphabet_sizes, strict=False)):
        if not (0 <= s < k):
            raise ValueError(
                f"Invalid state: state[{i}]={s} is not in [0, {k}) for "
                f"alphabet size {k}."
            )


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
    """Raise |StateUnreachableForwardsError| if the state is unreachable.

    Two checks fire:

    1. Substrate-level: the marginal probability
       ``P(state) = Σ_{s_t} ∏_i factor_i(s_t)[state[i]]`` must be positive
       under the substrate's joint factored TPM.
    2. Subsystem-level: the subsystem's component of the state must be
       producible by the background-conditioned subsystem dynamics. Some
       past state of the *subsystem* (with background fixed at the
       external state) must transition to the subsystem's ``proper_state``
       with nonzero probability.
    """
    factored = system.substrate.factored_tpm  # type: ignore[attr-defined]
    state = system.state  # type: ignore[attr-defined]
    pr_joint = np.ones(factored.alphabet_sizes, dtype=np.float64)
    for i in range(factored.n_nodes):
        pr_joint = pr_joint * factored.factor(i)[..., state[i]]
    if pr_joint.sum() <= 0.0:
        raise exceptions.StateUnreachableForwardsError(system.state)  # type: ignore[attr-defined]

    # Subsystem-level: conditioned dynamics must produce proper_state.
    if not _proper_state_in_image_of_conditioned_tpm(system):
        raise exceptions.StateUnreachableForwardsError(system.state)  # type: ignore[attr-defined]


def _proper_state_in_image_of_conditioned_tpm(system: object) -> bool:
    """Whether the subsystem's ``proper_state`` is in the image of the
    background-conditioned effect dynamics.

    ``proper_effect_marginal`` is a FactoredTPM with one factor per system
    output unit (background fixed at the external state, background input
    dims dropped). The state is in the image iff some system-input
    configuration assigns positive joint probability to ``proper_state`` —
    i.e. every system factor gives positive probability to its component
    of ``proper_state`` for that input. Works for any per-unit alphabet
    size.
    """
    proper = system.proper_effect_marginal  # type: ignore[attr-defined]
    proper_state = system.proper_state  # type: ignore[attr-defined]
    joint = np.ones(proper.alphabet_sizes, dtype=np.float64)
    for slot in range(proper.n_nodes):
        joint = joint * proper.factor(slot)[..., proper_state[slot]]
    return bool(np.any(joint > 0.0))


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
    node_states(s.state, s.substrate.factored_tpm.alphabet_sizes)  # type: ignore[attr-defined]
    system_partition(s.partition, s.partition_indices)  # type: ignore[attr-defined]
    if config.infrastructure.validate_system_states:
        state_reachable(s)
    return True


def relata(relata: Iterable[object] | None) -> None:
    """Validate a set of relata."""
    if not relata:
        raise ValueError("relata cannot be empty")
