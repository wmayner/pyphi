# validate.py
"""Methods for validating user input."""

from itertools import product
import numpy as np

from . import conf, exceptions
from .conf import config
from .direction import Direction


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
            "Input must be a Network (perhaps you passed a Subsystem instead?)"
        )


def node_states(state):
    """Check that the state contains only zeros and ones."""
    if not all(n in (0, 1) for n in state):
        raise ValueError("Invalid state: states must consist of only zeros and ones.")


def state_length(state, size):
    """Check that the state is the given size."""
    if len(state) != size:
        raise ValueError(
            "Invalid state: there must be one entry per "
            "node in the network; this state has {} entries, but "
            "there are {} nodes.".format(len(state), size)
        )
    return True


def state_type(state):
    """Check that the state only contains integers."""
    if any(not isinstance(s, int) for s in state):
        raise TypeError(
            f"Invalid state {state}: each entry must be of int type."
        )
    return True


def state_value(state, shape):
    """Check that each entry in the state falls within the right range."""
    if any(
            s not in range(cardinality)
            for s, cardinality in zip(state, shape)
    ):
        raise ValueError(
            f"Invalid state {state}: entries must be within zero and "
            f"{tuple((np.array(shape) - 1).tolist())}."
        )
    return True


def state(state, size, shape):
    """Check that the state is of the correct length, type and value."""
    return (
        state_length(state, size) and
        state_type(state) and
        state_value(state, shape)
    )

def _past_states(p_node):
    """Find set of states which could have led to the current state of a node.

    The state of irrelevant dimensions (nodes which don't output to this
    node) is represented with -1 to encode a whole equivalence class.

    Arguments:
        p_node (np.ndarray): Node TPM conditioned on the current subsystem state.
            See also :func:`pyphi.tpm.ImplicitTPM.probability_of_current_state`.

    Returns:
        set: Set of past states with nonzero probability of transitioning.
    """
    # Find s_{t-1} such that p_node > 0.
    states = list(np.argwhere(np.asarray(p_node) > 0))
    # Remove last dimension (probability of current state).
    states = [state[:-1] for state in states]
    # If node TPM shape at certain parent contains a 1, then
    # there's no dependency on that parent. Substitute '0' state
    # with placeholder -1 to encode equivalent states.
    states = [
        tuple(-1 if p_node.shape[i] == 1 else s for i, s in enumerate(state))
        for state in states
    ]
    return set(states)


def _states_intersection(states1, states2):
    """Efficient symbolic intersection between two sets of states.

    Arguments:
        states1 (set[tuple[int]]): First set of states or equivalence classes.
        states2 (set[tuple[int]]): Second set of states or equivalence classes.

    Returns:
        set[tuple[int]]: The intersection between the two sets.

    Examples:
        >>> states1 = {(1, 0, -1), (1, 1, 1)}
        >>> states2 = {(1, 0, 0), (1, 1, 1), (0, 0, 0)}
        >>> sorted(list(_states_intersection(states1, states2)))
        [(1, 0, 0), (1, 1, 1)]

        >>> states1 = {(1, -1, -1)}
        >>> states2 = {(1, 0, -1), (1, 1, -1)}
        >>> sorted(list(_states_intersection(states1, states2)))
        [(1, 0, -1), (1, 1, -1)]
    """
    def find_intersection(state_pair):
        # For each unordered pair |{state1, state2}| in the Cartesian product of
        # the two sets, check if |state1| and |state2| have a non-empty
        # (sub)class in common. If so, that is a member of the intersection.
        subclass = []
        for i, j in zip(*state_pair):
            if i == j:
                subclass.append(i)
            elif i == -1:
                subclass.append(j)
            elif j == -1:
                subclass.append(i)
            else:
                return None
        return tuple(subclass)

    # Lazy generator of the Cartesian product.
    state_pairs = product(states1, states2)
    # Find 2-ary intersections, filter out None's on the fly and return that set.
    return set(
        intersection for pair in state_pairs
        if (intersection := find_intersection(pair))
    )


def state_reachable(subsystem):
    """Raise exception if state cannot be reached according to subsystem's TPM."""
    # A state s is reachable by Subsystem S if and only if there is at least
    # one state s_{t-1} with nonzero probability of transitioning to s:
    #             ∃ s_{t-1} : p(s | s_{t-1}, w_{t-1}) > 0

    # Obtain p(s | w_{t-1}) as node marginals (i.e. implicitly).
    p = subsystem.proper_effect_tpm.probability_of_current_state(
        subsystem.proper_state
    )

    # Avoid computing the joint distribution. For each node n, find the set of
    # coordinates s_{t-1} for which p_n > 0. The intersection of all such sets
    # is the set of previous states leading to the current state.

    # Initial value.
    intersection = _past_states(p[0])

    for p_node in p[1:]:
        intersection = _states_intersection(intersection, _past_states(p_node))
        # Shortcircuit evaluation of intersection as soon as a
        # 2-ary intersection is empty.
        if not intersection:
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
    cut(s.cut, s.cut_indices)
    if config.VALIDATE_SUBSYSTEM_STATES:
        state_reachable(s)
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
