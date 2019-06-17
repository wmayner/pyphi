#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# validate.py

"""
Methods for validating arguments.
"""

import numpy as np

from . import Direction, config, convert, exceptions
from .tpm import is_state_by_state

# pylint: disable=redefined-outer-name


def direction(direction, allow_bi=False):
    """Validate that the given direction is one of the allowed constants.

    If ``allow_bi`` is ``True`` then ``Direction.BIDIRECTIONAL`` is
    acceptable.
    """
    valid = [Direction.CAUSE, Direction.EFFECT]
    if allow_bi:
        valid.append(Direction.BIDIRECTIONAL)

    if direction not in valid:
        raise ValueError('`direction` must be one of {}'.format(valid))

    return True


def tpm(tpm, check_independence=True):
    """Validate a TPM.

    The TPM can be in

        * 2-dimensional state-by-state form,
        * 2-dimensional state-by-node form, or
        * multidimensional state-by-node form.
    """
    see_tpm_docs = (
        'See the documentation on TPM conventions and the `pyphi.Network` '
        'object for more information on TPM forms.'
    )
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of nodes from the state-by-node TPM.
    N = tpm.shape[-1]
    if tpm.ndim == 2:
        if not ((tpm.shape[0] == 2**N and tpm.shape[1] == N) or
                (tpm.shape[0] == tpm.shape[1])):
            raise ValueError(
                'Invalid shape for 2-D TPM: {}\nFor a state-by-node TPM, '
                'there must be ' '2^N rows and N columns, where N is the '
                'number of nodes. State-by-state TPM must be square. '
                '{}'.format(tpm.shape, see_tpm_docs))
        if tpm.shape[0] == tpm.shape[1] and check_independence:
            conditionally_independent(tpm)
    elif tpm.ndim == (N + 1):
        if tpm.shape != tuple([2] * N + [N]):
            raise ValueError(
                'Invalid shape for multidimensional state-by-node TPM: {}\n'
                'The shape should be {} for {} nodes. {}'.format(
                    tpm.shape, ([2] * N) + [N], N, see_tpm_docs))
    else:
        raise ValueError(
            'Invalid TPM: Must be either 2-dimensional or multidimensional. '
            '{}'.format(see_tpm_docs))
    return True


def conditionally_independent(tpm):
    """Validate that the TPM is conditionally independent."""
    tpm = np.array(tpm)
    if is_state_by_state(tpm):
        there_and_back_again = convert.state_by_node2state_by_state(
            convert.state_by_state2state_by_node(tpm))
    else:
        there_and_back_again = convert.state_by_state2state_by_node(
            convert.state_by_node2state_by_state(tpm))
    if not np.allclose((tpm - there_and_back_again), 0.0):
        raise exceptions.ConditionallyDependentError(
            'TPM is not conditionally independent.\n'
            'See the conditional independence example in the documentation '
            'for more info.')
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
        raise ValueError("Connectivity matrix must contain only binary "
                         "values.")
    return True


def node_labels(node_labels, node_indices):
    """Validate that there is a label for each node."""
    if len(node_labels) != len(node_indices):
        raise ValueError("Labels {0} must label every node {1}.".format(
            node_labels, node_indices))

    if len(node_labels) != len(set(node_labels)):
        raise ValueError("Labels {0} must be unique.".format(node_labels))


def network(n):
    """Validate a |Network|.

    Checks the TPM and connectivity matrix.
    """
    tpm(n.tpm)
    connectivity_matrix(n.cm)
    if n.cm.shape[0] != n.size:
        raise ValueError("Connectivity matrix must be NxN, where N is the "
                         "number of nodes in the network.")
    return True


def is_network(network):
    """Validate that the argument is a |Network|."""
    from . import Network

    if not isinstance(network, Network):
        raise ValueError(
            "Input must be a Network (perhaps you passed a Subsystem instead?")


def node_states(state):
    """Check that the state contains only zeros and ones."""
    if not all(n in (0, 1) for n in state):
        raise ValueError(
            'Invalid state: states must consist of only zeros and ones.')


def state_length(state, size):
    """Check that the state is the given size."""
    if len(state) != size:
        raise ValueError('Invalid state: there must be one entry per '
                         'node in the network; this state has {} entries, but '
                         'there are {} nodes.'.format(len(state), size))
    return True


def state_reachable(subsystem):
    """Return whether a state can be reached according to the network's TPM."""
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    # First we take the submatrix of the conditioned TPM that corresponds to
    # the nodes that are actually in the subsystem...
    tpm = subsystem.tpm[..., subsystem.node_indices]
    # Then we do the subtraction and test.
    test = tpm - np.array(subsystem.proper_state)
    if not np.any(np.logical_and(-1 < test, test < 1).all(-1)):
        raise exceptions.StateUnreachableError(subsystem.state)


def cut(cut, node_indices):
    """Check that the cut is for only the given nodes."""
    if cut.indices != node_indices:
        raise ValueError('{} nodes are not equal to subsystem nodes '
                         '{}'.format(cut, node_indices))


def subsystem(s):
    """Validate a |Subsystem|.

    Checks its state and cut.
    """
    node_states(s.state)
    cut(s.cut, s.cut_indices)
    if config.VALIDATE_SUBSYSTEM_STATES:
        state_reachable(s)
    return True


def time_scale(time_scale):
    """Validate a macro temporal time scale."""
    if time_scale <= 0 or isinstance(time_scale, float):
        raise ValueError('time scale must be a positive integer')


def partition(partition):
    """Validate a partition - used by blackboxes and coarse grains."""
    nodes = set()
    for part in partition:
        for node in part:
            if node in nodes:
                raise ValueError(
                    'Micro-element {} may not be partitioned into multiple '
                    'macro-elements'.format(node))
            nodes.add(node)


def coarse_grain(coarse_grain):
    """Validate a macro coarse-graining."""
    partition(coarse_grain.partition)

    if len(coarse_grain.partition) != len(coarse_grain.grouping):
        raise ValueError('output and state groupings must be the same size')

    for part, group in zip(coarse_grain.partition, coarse_grain.grouping):
        if set(range(len(part) + 1)) != set(group[0] + group[1]):
            # Check that all elements in the partition are in one of the two
            # state groupings
            raise ValueError('elements in output grouping {0} do not match '
                             'elements in state grouping {1}'.format(
                                 part, group))


def blackbox(blackbox):
    """Validate a macro blackboxing."""
    if tuple(sorted(blackbox.output_indices)) != blackbox.output_indices:
        raise ValueError('Output indices {} must be ordered'.format(
            blackbox.output_indices))

    partition(blackbox.partition)

    for part in blackbox.partition:
        if not set(part) & set(blackbox.output_indices):
            raise ValueError(
                'Every blackbox must have an output - {} does not'.format(
                    part))


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
                'A blackboxing with multiple outputs per box must be '
                'coarse-grained.')

        if (coarse_grain and not any(outputs.issubset(part)
                                     for part in coarse_grain.partition)):
            raise ValueError(
                'Multiple outputs from a blackbox must be partitioned into '
                'the same macro-element of the coarse-graining')
