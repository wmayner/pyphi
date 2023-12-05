# validate.py

"""
Methods for validating arguments.
"""

from  warnings import warn

import numpy as np

from . import exceptions
from .conf import config
from .direction import Direction
from .tpm import ImplicitTPM, reconstitute_tpm


# pylint: disable=redefined-outer-name


# TODO(4.0) move to `Direction`
def directions(directions, **kwargs):
    return all(direction(d, **kwargs) for d in directions)


def direction(direction, allow_bi=False):
    """Validate that the given direction is one of the allowed constants.

    If ``allow_bi`` is ``True`` then ``Direction.BIDIRECTIONAL`` is
    acceptable.
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


def node_labels(node_labels, node_indices):
    """Validate that there is a label for each node."""
    if len(node_labels) != len(node_indices):
        raise ValueError(
            "Labels {0} must label every node {1}.".format(node_labels, node_indices)
        )

    if len(node_labels) != len(set(node_labels)):
        raise ValueError("Labels {0} must be unique.".format(node_labels))

def network(n):
    """Validate a |Network|.

    Checks the TPM and connectivity matrix.
    """
    n.tpm.validate()
    connectivity_matrix(n.cm)
    shapes(n.tpm.shapes, n.cm)
    if n.cm.shape[0] != n.size:
        raise ValueError(
            "Connectivity matrix must be NxN, where N is the "
            "number of nodes in the network."
        )
    return True


def connectivity_matrix(cm):
    """Validate the given connectivity matrix."""
    # Special case for empty matrices.
    if cm.size == 0:
        return True
    if cm.ndim != 2:
        raise ValueError("Connectivity matrix must be 2-dimensional.")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if not np.all(np.logical_or(cm == 1, cm == 0)):
        raise ValueError("Connectivity matrix must contain only binary " "values.")
    return True


def shapes(shapes, cm):
    """Validate consistency between node TPM shapes and a user-provided cm."""
    for i, shape in enumerate(shapes):
        for j, con in enumerate(cm[..., i]):
            if (con == 0 and shape[j] != 1) or (con != 0 and shape[j] == 1):
                raise ValueError(
                    "Node TPM {} of shape {} does not match the connectivity "
                    "matrix.".format(i, shape)
                )
    return True


def is_network(network):
    """Validate that the argument is a |Network|."""
    from . import Network

    if not isinstance(network, Network):
        raise ValueError(
            "Input must be a Network (perhaps you passed a Subsystem instead?"
        )


def state_length(state, size):
    """Check that the state is the given size."""
    if len(state) != size:
        raise ValueError(
            "Invalid state: there must be one entry per "
            "node in the network; this state has {} entries, but "
            "there are {} nodes.".format(len(state), size)
        )
    return True


def state_reachable(subsystem):
    """Return whether a state can be reached according to the network's TPM."""
    # TODO(tpm) Change consumers of this function, so that only ImplicitTPMs
    # are passed.
    tpm = (
        reconstitute_tpm(subsystem.tpm) if isinstance(subsystem.tpm, ImplicitTPM)
        else subsystem.tpm
    )
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    # First we take the submatrix of the conditioned TPM that corresponds to
    # the nodes that are actually in the subsystem...
    tpm = tpm[..., subsystem.node_indices]
    # Make sure the state is translated in terms of integer indices.
    # TODO(tpm) Simplify conversion with a state_space class?
    state_space = [
        node.state_space for node in subsystem.nodes
        if node.index in subsystem.node_indices
    ]
    state = np.array([
        state_space[node].index(state)
        for node, state in enumerate(subsystem.proper_state)
    ])
    # Then we do the subtraction and test.
    test = tpm - state
    if not np.any(np.logical_and(-1 < test, test < 1).all(-1)):
        raise exceptions.StateUnreachableError(subsystem.state)


def cut(cut, node_indices):
    """Check that the cut is for only the given nodes."""
    if set(cut.indices) != set(node_indices):
        raise ValueError(
            "{} nodes are not equal to subsystem nodes " "{}".format(cut, node_indices)
        )


def subsystem(s):
    """Validate a |Subsystem|.

    Checks its state and cut.
    """
    # cut(s.cut, s.cut_indices)
    if config.VALIDATE_SUBSYSTEM_STATES:
        # TODO(tpm) Reimplement in a way that never reconstitutes the full TPM.
        # state_reachable(s)
        # warn("Validation of state reachability didn't take place.")
        pass
    return True


def time_scale(time_scale):
    """Validate a macro temporal time scale."""
    if time_scale <= 0 or isinstance(time_scale, float):
        raise ValueError("time scale must be a positive integer")


def partition(partition):
    """Validate a partition - used by blackboxes and coarse grains."""
    nodes = set()
    for part in partition:
        for node in part:
            if node in nodes:
                raise ValueError(
                    "Micro-element {} may not be partitioned into multiple "
                    "macro-elements".format(node)
                )
            nodes.add(node)


def coarse_grain(coarse_grain):
    """Validate a macro coarse-graining."""
    partition(coarse_grain.partition)

    if len(coarse_grain.partition) != len(coarse_grain.grouping):
        raise ValueError("output and state groupings must be the same size")

    for part, group in zip(coarse_grain.partition, coarse_grain.grouping):
        if set(range(len(part) + 1)) != set(group[0] + group[1]):
            # Check that all elements in the partition are in one of the two
            # state groupings
            raise ValueError(
                "elements in output grouping {0} do not match "
                "elements in state grouping {1}".format(part, group)
            )


def blackbox(blackbox):
    """Validate a macro blackboxing."""
    if tuple(sorted(blackbox.output_indices)) != blackbox.output_indices:
        raise ValueError(
            "Output indices {} must be ordered".format(blackbox.output_indices)
        )

    partition(blackbox.partition)

    for part in blackbox.partition:
        if not set(part) & set(blackbox.output_indices):
            raise ValueError(
                "Every blackbox must have an output - {} does not".format(part)
            )


def blackbox_and_coarse_grain(blackbox, coarse_grain):
    """Validate that a coarse-graining properly combines the outputs of a
    blackboxing.
    """
    if blackbox is None:
        return

    for box in blackbox.partition:
        # Outputs of the box
        outputs = set(box) & set(blackbox.output_indices)

        if coarse_grain is None and len(outputs) > 1:
            raise ValueError(
                "A blackboxing with multiple outputs per box must be " "coarse-grained."
            )

        if coarse_grain and not any(
            outputs.issubset(part) for part in coarse_grain.partition
        ):
            raise ValueError(
                "Multiple outputs from a blackbox must be partitioned into "
                "the same macro-element of the coarse-graining"
            )


def relata(relata):
    """Validate a set of relata."""
    if not relata:
        raise ValueError("relata cannot be empty")
