#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convert.py

'''
Conversion functions.

See the documentation on PyPhi |conventions| for information on the different
representations that these functions convert between.
'''

import logging
from math import log2

import numpy as np

# Create a logger for this module.
log = logging.getLogger(__name__)


def reverse_bits(i, n):
    '''Reverse the bits of the ``n``-bit decimal number ``i``.

    Examples:
        >>> reverse_bits(12, 7)
        24
        >>> reverse_bits(0, 1)
        0
        >>> reverse_bits(1, 2)
        2
    '''
    return int(bin(i)[2:].zfill(n)[::-1], 2)


def nodes2indices(nodes):
    '''Convert nodes to a tuple of their indices.'''
    return tuple(n.index for n in nodes) if nodes else ()


def nodes2state(nodes):
    '''Convert nodes to a tuple of their states.'''
    return tuple(n.state for n in nodes) if nodes else ()


def holi2loli(i, n):
    '''Convert between HOLI and LOLI for indices in ``range(n)``.'''
    return reverse_bits(i, n)


loli2holi = holi2loli


def state2holi_index(state):
    '''Convert a PyPhi state-tuple to a decimal index according to the HOLI
    convention.

    Args:
        state (tuple[int]): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        int: A decimal integer corresponding to a network state under the
        HOLI convention.

    Examples:
        >>> state2holi_index((1, 0, 0, 0, 0))
        16
        >>> state2holi_index((1, 1, 1, 0, 0, 0, 0, 0))
        224
    '''
    return int(''.join(str(int(n)) for n in state), 2)


def state2loli_index(state):
    '''Convert a PyPhi state-tuple to a decimal index according to the LOLI
    convention.

    Args:
        state (tuple[int]): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        int: A decimal integer corresponding to a network state under the
        LOLI convention.

    Examples:
        >>> state2loli_index((1, 0, 0, 0, 0))
        1
        >>> state2loli_index((1, 1, 1, 0, 0, 0, 0, 0))
        7
    '''
    return int(''.join(str(int(n)) for n in state[::-1]), 2)


def loli_index2state(i, number_of_nodes):
    '''Convert a decimal integer to a PyPhi state tuple with the LOLI
    convention.

    The output is the reverse of |holi_index2state|.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            LOLI convention.

    Returns:
        tuple[int]: A state-tuple where the |ith| element of the tuple gives
        the state of the |ith| node.

    Examples:
        >>> number_of_nodes = 5
        >>> loli_index2state(1, number_of_nodes)
        (1, 0, 0, 0, 0)
        >>> number_of_nodes = 8
        >>> loli_index2state(7, number_of_nodes)
        (1, 1, 1, 0, 0, 0, 0, 0)
    '''
    return tuple((i >> n) & 1 for n in range(number_of_nodes))


def holi_index2state(i, number_of_nodes):
    '''Convert a decimal integer to a PyPhi state tuple using the HOLI
    convention that high-order bits correspond to low-index nodes.

    The output is the reverse of |loli_index2state|.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            HOLI convention.

    Returns:
        tuple[int]: A state-tuple where the |ith| element of the tuple gives
        the state of the |ith| node.

    Examples:
        >>> number_of_nodes = 5
        >>> holi_index2state(1, number_of_nodes)
        (0, 0, 0, 0, 1)
        >>> number_of_nodes = 8
        >>> holi_index2state(7, number_of_nodes)
        (0, 0, 0, 0, 0, 1, 1, 1)
    '''
    return loli_index2state(i, number_of_nodes)[::-1]


def holi2loli_state_by_state(tpm):
    '''Convert a state-by-state TPM from HOLI to LOLI or vice versa.

    Args:
        tpm (np.ndarray): A state-by-state TPM.

    Returns:
        np.ndarray: The state-by-state TPM in the other indexing format.

    Example:
        >>> tpm = np.arange(16).reshape([4, 4])
        >>> holi2loli_state_by_state(tpm)
        array([[  0.,   1.,   2.,   3.],
               [  8.,   9.,  10.,  11.],
               [  4.,   5.,   6.,   7.],
               [ 12.,  13.,  14.,  15.]])
    '''
    loli = np.empty(tpm.shape)
    N = tpm.shape[0]
    n = int(log2(N))
    for i in range(N):
        loli[i, :] = tpm[holi2loli(i, n), :]
    return loli


loli2holi_state_by_state = holi2loli_state_by_state


def to_n_dimensional(tpm):
    '''Reshape a state-by-node TPM to the n-dimensional form.

    See documentation for the |Network| object for more information on TPM
    formats.
    '''
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of nodes.
    N = tpm.shape[-1]
    # Reshape. We use Fortran ordering here so that the rows use the LOLI
    # convention (low-order bits correspond to low-index nodes). Note that this
    # does not change the actual memory layout (C- or Fortran-contiguous), so
    # there is no performance loss.
    return tpm.reshape([2] * N + [N], order="F").astype(float)


def to_2_dimensional(tpm):
    '''Reshape a state-by-node TPM to the 2-dimensional form.

    See documentation for the |Network| object for more information on TPM
    formats.
    '''
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of nodes.
    N = tpm.shape[-1]
    # Reshape.
    return tpm.reshape([2**N, N], order="F").astype(float)


def state_by_state2state_by_node(tpm):
    '''Convert a state-by-state TPM to a state-by-node TPM.

    .. danger::
        Many nondeterministic state-by-state TPMs can be represented by a
        single a state-by-state TPM. However, the mapping can be made to be
        one-to-one if we assume the state-by-state TPM is conditionally
        independent, as this function does. **If the given TPM is not
        conditionally independent, the conditional dependencies will be
        silently lost.**

    .. note::
        The indices of the rows and columns of the state-by-state TPM are
        assumed to follow the LOLI convention. The indices of the rows of the
        resulting state-by-node TPM also follow the LOLI convention. See the
        documentation on PyPhi |conventions| more information.

    Args:
        tpm (list[list] or np.ndarray): A square state-by-state TPM with row
            and column indices following the LOLI convention.

    Returns:
        np.ndarray: A state-by-node TPM, with row indices following the
        LOLI convention.

    Example:
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
    '''
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of states from the length of one side of the TPM.
    S = tpm.shape[-1]
    # Get the number of nodes from the number of states.
    N = int(log2(S))
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
    return sbn_tpm


# TODO support nondeterministic TPMs
# TODO add documentation on TPM representation and conditional independence and
# reference it here
def state_by_node2state_by_state(tpm):
    '''Convert a state-by-node TPM to a state-by-state TPM.

    .. important::
        A nondeterministic state-by-node TPM can have more than one
        representation as a state-by-state TPM. However, the mapping can be
        made to be one-to-one if we assume the TPMs to be conditionally
        independent. Therefore, **this function returns the corresponding
        conditionally independent state-by-state TPM.**

    .. note::
        The indices of the rows of the state-by-node TPM are assumed to follow
        the LOLI convention, while the indices of the columns follow the HOLI
        convention. The indices of the rows and columns of the resulting
        state-by-state TPM both follow the HOLI convention. See the
        documentation on PyPhi |conventions| for more info.

    Args:
        tpm (list[list] or np.ndarray): A state-by-node TPM with row indices
            following the LOLI convention and column indices following the HOLI
            convention.

    Returns:
        np.ndarray: A state-by-state TPM, with both row and column indices
        following the HOLI convention.

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
    '''
    # Cast to np.array.
    tpm = np.array(tpm)
    # Convert to n-dimensional form.
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


# Short aliases

h2l = holi2loli
l2h = loli2holi
l2s = loli_index2state
h2s = holi_index2state
s2l = state2loli_index
s2h = state2holi_index
h2l_sbs = holi2loli_state_by_state
l2h_sbs = loli2holi_state_by_state
to_n_d = to_n_dimensional
to_2_d = to_2_dimensional
sbn2sbs = state_by_node2state_by_state
sbs2sbn = state_by_state2state_by_node
