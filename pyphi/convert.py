#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# convert.py
"""
Conversion functions.
"""

import math
import numpy as np
from . import validate
import logging
from .constants import EPSILON


# Create a logger for this module.
log = logging.getLogger(__name__)


def nodes2indices(nodes):
    return tuple(n.index for n in nodes)


def state2holi_index(state):
    """Convert a PyPhi state-tuple to a decimal index according to the **HOLI**
    convention.

    Args:
        state (tuple(int)): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        ``int`` -- A decimal integer corresponding to a network state under the
            **HOLI** convention.

    Examples:
        >>> from pyphi.convert import state2loli_index
        >>> state2holi_index((1, 0, 0, 0, 0))
        16
        >>> state2holi_index((1, 1, 1, 0, 0, 0, 0, 0))
        224
    """
    return int(''.join(str(int(n)) for n in state), 2)


def state2loli_index(state):
    """Convert a PyPhi state-tuple to a decimal index according to the **LOLI**
    convention.

    Args:
        state (tuple(int)): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        ``int`` -- A decimal integer corresponding to a network state under the
            **LOLI** convention.

    Examples:
        >>> from pyphi.convert import state2loli_index
        >>> state2loli_index((1, 0, 0, 0, 0))
        1
        >>> state2loli_index((1, 1, 1, 0, 0, 0, 0, 0))
        7
    """
    return int(''.join(str(int(n)) for n in state[::-1]), 2)


def loli_index2state(i, number_of_nodes):
    """Convert a decimal integer to a PyPhi state tuple with the **LOLI**
    convention.

    The output is the reverse of :func:`holi_index2state`.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            **LOLI** convention.

    Returns:
        ``tuple(int)`` -- A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Examples:
        >>> from pyphi.convert import loli_index2state
        >>> number_of_nodes = 5
        >>> loli_index2state(1, number_of_nodes)
        (1, 0, 0, 0, 0)
        >>> number_of_nodes = 8
        >>> loli_index2state(7, number_of_nodes)
        (1, 1, 1, 0, 0, 0, 0, 0)
    """
    return tuple((i >> n) & 1 for n in range(number_of_nodes))


def holi_index2state(i, number_of_nodes):
    """Convert a decimal integer to a PyPhi state tuple using the **HOLI**
    convention that high-order bits correspond to low-index nodes.

    The output is the reverse of :func:`loli_index2state`.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            **HOLI** convention.

    Returns:
        ``tuple(int)`` -- A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Examples:
        >>> from pyphi.convert import holi_index2state
        >>> number_of_nodes = 5
        >>> holi_index2state(1, number_of_nodes)
        (0, 0, 0, 0, 1)
        >>> number_of_nodes = 8
        >>> holi_index2state(7, number_of_nodes)
        (0, 0, 0, 0, 0, 1, 1, 1)
    """
    return loli_index2state(i, number_of_nodes)[::-1]


def to_n_dimensional(tpm):
    """Reshape a state-by-node TPM to the N-D form.

    See documentation for :class:`pyphi.network` for more information on TPM
    formats."""
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of nodes.
    N = tpm.shape[-1]
    # Reshape. We use Fortran ordering here so that the rows use the LOLI
    # convention (low-order bits correspond to low-index nodes). Note that this
    # does not change the actual memory layout (C- or Fortran-contiguous), so
    # there is no performance loss.
    return tpm.reshape([2] * N + [N], order="F").astype(float)


def state_by_state2state_by_node(tpm):
    """Convert a state-by-state TPM to a state-by-node TPM.

    .. note::
        The indices of the rows and columns of the state-by-state TPM are
        assumed to follow the **LOLI** convention. The indices of the rows of
        the resulting state-by-node TPM also follow the **LOLI** convention.
        See the documentation for :class:`pyphi.examples` for more info on
        these conventions.

    Args:
        tpm (list(list) or np.ndarray): A square state-by-state TPM with row
            and column indices following the **LOLI** convention.

    Returns:
        ``np.ndarray`` -- A state-by-node TPM, with row indices following the
            **LOLI** convention.

    Examples:
        >>> from pyphi.convert import state_by_state2state_by_node
        >>> tpm = np.array([[0.5, 0.5, 0.0, 0.0],
        ...                 [0.0, 1.0, 0.0, 0.0],
        ...                 [0.0, 0.2, 0.0, 0.8],
        ...                 [0.0, 0.3, 0.7, 0.0]])
        >>> state_by_state2state_by_node(tpm)
        array([[[ 0.5,  0. ],
                [ 1. ,  0.8]],
        <BLANKLINE>
               [[ 1. ,  0. ],
                [ 0.3,  0.7]]])
    """
    # Validate the TPM.
    validate.tpm(tpm)
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of states from the length of one side of the TPM.
    S = tpm.shape[-1]
    # Get the number of nodes from the number of states.
    N = int(math.log(S, 2))
    # Initialize the new state-by node TPM.
    sbn_tpm = np.zeros(([2] * N + [N]))
    # Map indices to state-tuples with the LOLI convention.
    states = {i: loli_index2state(i, N) for i in range(S)}
    # Get an array for each node with 1 in positions that correspond to that
    # node being on in the next state, and a 0 otherwise.
    node_on = np.array([[states[i][n] for i in range(S)] for n in range(N)])
    on_probabilities = [tpm * node_on[n] for n in range(N)]
    for i, state in states.items():
        # Get the probability of each node being on given the past state i,
        # i.e., a row of the state-by-node TPM.
        # Assign that row to the ith state in the state-by-node TPM.
        sbn_tpm[state] = [np.sum(on_probabilities[n][i]) for n in range(N)]
    if not np.all((tpm - state_by_node2state_by_state(sbn_tpm)) < EPSILON):
        logging.warning(
            'The TPM is not conditionally independent. See the conditional '
            'independence example in the documentation for more information '
            'on how this is handled.')
    return sbn_tpm


# TODO support nondeterministic TPMs
def state_by_node2state_by_state(tpm):
    """Convert a state-by-node TPM to a state-by-state TPM.

    .. note::
        **A nondeterministic state-by-node TPM can have more than one
        representation as a state-by-state TPM.** However, the mapping can be
        made to be one-to-one if we assume the TPMs to be conditionally
        independent. Therefore, given a nondeterministic state-by-node TPM,
        this function returns the corresponding conditionally independent
        state-by-state.

    .. note::
        The indices of the rows of the state-by-node TPM are assumed to follow
        the **LOLI** convention, while the indices of the columns follow the
        **HOLI** convention. The indices of the rows and columns of the
        resulting state-by-state TPM both follow the **HOLI** convention.

    Args:
        tpm (list(list) or np.ndarray): A state-by-node TPM with row indices
            following the **LOLI** convention and column indices following the
            **HOLI** convention.

    Returns:
        ``np.ndarray`` -- A state-by-state TPM, with both row and column
            indices following the **HOLI** convention.

    >>> from pyphi.convert import state_by_node2state_by_state
    >>> tpm = np.array([[1, 1, 0],
    ...                 [0, 0, 1],
    ...                 [0, 1, 1],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [1, 1, 1],
    ...                 [1, 0, 1]])
    >>> state_by_node2state_by_state(tpm)
    array([[ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]])
    """
    # Cast to np.array.
    tpm = np.array(tpm)
    # Validate the TPM.
    validate.tpm(tpm)
    # Convert to N-D form.
    tpm = to_n_dimensional(tpm)
    # Get the number of nodes from the last dimension of the TPM.
    N = tpm.shape[-1]
    # Get the number of states.
    S = 2**N
    # Initialize the state-by-state TPM.
    sbs_tpm = np.zeros((S, S))
    if not np.any(np.logical_and(tpm < 1, tpm > 0)):
        # TPM is deterministic.
        for past_state_index in range(S):
            # Use the LOLI convention to get the row and column indices.
            past_state = loli_index2state(past_state_index, N)
            current_state_index = state2loli_index(tpm[past_state])
            sbs_tpm[past_state_index, current_state_index] = 1
    else:
        # TPM is nondeterministic.
        for past_state_index in range(S):
            # Use the LOLI convention to get the row and column indices.
            past_state = loli_index2state(past_state_index, N)
            marginal_tpm = tpm[past_state]
            for current_state_index in range(S):
                current_state = np.array(
                    [i for i in loli_index2state(current_state_index, N)])
                sbs_tpm[past_state_index, current_state_index] = (
                    np.prod(marginal_tpm[current_state == 1]) *
                    np.prod(1 - marginal_tpm[current_state == 0]))
    return sbs_tpm
