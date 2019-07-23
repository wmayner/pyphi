#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convert.py

"""
Conversion functions.

See the documentation on PyPhi :ref:`tpm-conventions` for information on the
different representations that these functions convert between.
"""

import logging
from math import log2

import numpy as np

from .tpm import is_deterministic

# Create a logger for this module.
log = logging.getLogger(__name__)


def reverse_bits(i, n):
    """Reverse the bits of the ``n``-bit decimal number ``i``.

    Examples:
        >>> reverse_bits(12, 7)
        24
        >>> reverse_bits(0, 1)
        0
        >>> reverse_bits(1, 2)
        2
    """
    return int(bin(i)[2:].zfill(n)[::-1], 2)


def be2le(i, n):
    """Convert between big-endian and little-endian for indices in
    ``range(n)``.
    """
    return reverse_bits(i, n)


le2be = be2le


def nodes2indices(nodes):
    """Convert nodes to a tuple of their indices."""
    return tuple(n.index for n in nodes) if nodes else ()


def nodes2state(nodes):
    """Convert nodes to a tuple of their states."""
    return tuple(n.state for n in nodes) if nodes else ()


def state2be_index(state):
    """Convert a PyPhi state-tuple to a decimal index according to the
    big-endian convention.

    Args:
        state (tuple[int]): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        int: A decimal integer corresponding to a network state under the
        big-endian convention.

    Examples:
        >>> state2be_index((1, 0, 0, 0, 0))
        16
        >>> state2be_index((1, 1, 1, 0, 0, 0, 0, 0))
        224
    """
    return int(''.join(str(int(n)) for n in state), 2)


def state2le_index(state):
    """Convert a PyPhi state-tuple to a decimal index according to the
    little-endian convention.

    Args:
        state (tuple[int]): A state-tuple where the |ith| element of the tuple
            gives the state of the |ith| node.

    Returns:
        int: A decimal integer corresponding to a network state under the
        little-endian convention.

    Examples:
        >>> state2le_index((1, 0, 0, 0, 0))
        1
        >>> state2le_index((1, 1, 1, 0, 0, 0, 0, 0))
        7
    """
    return int(''.join(str(int(n)) for n in state[::-1]), 2)


def le_index2state(i, number_of_nodes):
    """Convert a decimal integer to a PyPhi state tuple with the little-endian
    convention.

    The output is the reverse of |be_index2state()|.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            little-endian convention.

    Returns:
        tuple[int]: A state-tuple where the |ith| element of the tuple gives
        the state of the |ith| node.

    Examples:
        >>> number_of_nodes = 5
        >>> le_index2state(1, number_of_nodes)
        (1, 0, 0, 0, 0)
        >>> number_of_nodes = 8
        >>> le_index2state(7, number_of_nodes)
        (1, 1, 1, 0, 0, 0, 0, 0)
    """
    return tuple((i >> n) & 1 for n in range(number_of_nodes))


def be_index2state(i, number_of_nodes):
    """Convert a decimal integer to a PyPhi state tuple using the big-endian
    convention that the most-significant bits correspond to low-index nodes.

    The output is the reverse of |le_index2state()|.

    Args:
        i (int): A decimal integer corresponding to a network state under the
            big-endian convention.

    Returns:
        tuple[int]: A state-tuple where the |ith| element of the tuple gives
        the state of the |ith| node.

    Examples:
        >>> number_of_nodes = 5
        >>> be_index2state(1, number_of_nodes)
        (0, 0, 0, 0, 1)
        >>> number_of_nodes = 8
        >>> be_index2state(7, number_of_nodes)
        (0, 0, 0, 0, 0, 1, 1, 1)
    """
    return le_index2state(i, number_of_nodes)[::-1]


def be2le_state_by_state(tpm):
    """Convert a state-by-state TPM from big-endian to little-endian or vice
    versa.

    Args:
        tpm (np.ndarray): A state-by-state TPM.

    Returns:
        np.ndarray: The state-by-state TPM in the other indexing format.

    Example:
        >>> tpm = np.arange(16).reshape([4, 4])
        >>> be2le_state_by_state(tpm)
        array([[ 0.,  1.,  2.,  3.],
               [ 8.,  9., 10., 11.],
               [ 4.,  5.,  6.,  7.],
               [12., 13., 14., 15.]])
    """
    le = np.empty(tpm.shape)
    N = tpm.shape[0]
    n = int(log2(N))
    for i in range(N):
        le[i, :] = tpm[be2le(i, n), :]
    return le


le2be_state_by_state = be2le_state_by_state


def to_multidimensional(tpm):
    """Reshape a state-by-node TPM to the multidimensional form.

    See documentation for the |Network| object for more information on TPM
    formats.
    """
    # Cast to np.array
    tpm = np.array(tpm)
    # Get the number of nodes in the previous state
    Np = int(log2(np.prod(tpm.shape[:-1])))
    # Get the number of nodes in the next state
    Nn = tpm.shape[-1]
    # Reshape. We use Fortran ordering here so that the rows use the
    # little-endian convention (least-significant bits correspond to low-index
    # nodes). Note that this does not change the actual memory layout (C- or
    # Fortran-contiguous), so there is no performance loss.
    return tpm.reshape([2] * Np + [Nn], order="F").astype(float)


def to_2dimensional(tpm):
    """Reshape a state-by-node TPM to the 2-dimensional form.

    See :ref:`tpm-conventions` and documentation for the |Network| object for
    more information on TPM representations.
    """
    # Cast to np.array
    tpm = np.array(tpm)
    # Get the number of previous states
    S = np.prod(tpm.shape[:-1])
    # Get the number of next states
    N = tpm.shape[-1]
    # Reshape
    return tpm.reshape([S, N], order="F").astype(float)


def state_by_state2state_by_node(tpm):
    """Convert a state-by-state TPM to a state-by-node TPM.

    .. danger::
        Many nondeterministic state-by-state TPMs can be represented by a
        single a state-by-state TPM. However, the mapping can be made to be
        one-to-one if we assume the state-by-state TPM is conditionally
        independent, as this function does. **If the given TPM is not
        conditionally independent, the conditional dependencies will be
        silently lost.**

    .. note::
        The indices of the rows and columns of the state-by-state TPM are
        assumed to follow the little-endian convention. The indices of the rows
        of the resulting state-by-node TPM also follow the little-endian
        convention. See the documentation on PyPhi the :ref:`tpm-conventions`
        more information.

    Args:
        tpm (list[list] or np.ndarray): A square state-by-state TPM with row
            and column indices following the little-endian convention.

    Returns:
        np.ndarray: A state-by-node TPM, with row indices following the
        little-endian convention.

    Example:
        >>> tpm = np.array([[0.5, 0.5, 0.0, 0.0],
        ...                 [0.0, 1.0, 0.0, 0.0],
        ...                 [0.0, 0.2, 0.0, 0.8],
        ...                 [0.0, 0.3, 0.7, 0.0]])
        >>> state_by_state2state_by_node(tpm)
        array([[[0.5, 0. ],
                [1. , 0.8]],
        <BLANKLINE>
               [[1. , 0. ],
                [0.3, 0.7]]])
    """
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of states from the length of one side of the TPM.
    S = tpm.shape[-1]
    # Get the number of nodes from the number of states.
    N = int(log2(S))
    # Initialize the new state-by node TPM.
    sbn_tpm = np.zeros(([2] * N + [N]))
    # Map indices to state-tuples with the little-endian convention.
    states = {i: le_index2state(i, N) for i in range(S)}
    # Get an array for each node with 1 in positions that correspond to that
    # node being on in the next state, and a 0 otherwise.
    node_on = np.array([[states[i][n] for i in range(S)] for n in range(N)])
    on_probabilities = [tpm * node_on[n] for n in range(N)]
    for i, state in states.items():
        # Get the probability of each node being on given the previous state i,
        # i.e., a row of the state-by-node TPM.
        # Assign that row to the ith state in the state-by-node TPM.
        sbn_tpm[state] = [np.sum(on_probabilities[n][i]) for n in range(N)]
    return sbn_tpm


def _deterministic_sbn2sbs(Sp, Sn, sbn_tpm):
    # Initialize the state-by-state TPM
    sbs_tpm = np.zeros((Sp, Sn))
    for previous_state_index in range(Sp):
        # Use the little-endian convention to get the row and column
        # indices
        current_state_index = state2le_index(sbn_tpm[previous_state_index])
        sbs_tpm[previous_state_index, current_state_index] = 1
    return sbs_tpm


def _unfold_nodewise_probabilities(Nn, Sn, sbn_row):
    # We make the probabilites associated with each state of the N nodes
    # explicit by generating all combinations of ON/OFF and then subtracting
    # the ON probability from 1 where the OFF probability is needed

    # Generate the ON/OFF combinations (we reverse the binary representation to
    # respect the little-endian convention)
    combinations = np.flip(np.array([
        list(np.binary_repr(i, width=Nn))
        for i in range(Sn)
    ], dtype=int), axis=1)
    # Replicate the row for each combination
    row_replicates = np.tile(sbn_row, (Sn, 1))
    # We take the absolute value to keep ON probabilities, which were
    # subtracted from 0, positive
    return np.abs(combinations - row_replicates)


def _nondeterministic_sbn2sbs(Nn, Sn, sbn_tpm):
    # Unfold the state-by-node probabilities
    unfolded = np.array([
        _unfold_nodewise_probabilities(Nn, Sn, row)
        for row in sbn_tpm
    ])
    # Now we take the product of the individual node probabilities to get the
    # probability of each state as a whole, flipping to achieve the ordering
    # implied by the little-endian convention
    return np.flip(np.prod(unfolded, axis=-1), axis=-1)


def state_by_node2state_by_state(tpm):
    """Convert a state-by-node TPM to a state-by-state TPM.

    .. important::
        A nondeterministic state-by-node TPM can have more than one
        representation as a state-by-state TPM. However, the mapping can be
        made to be one-to-one if we assume the TPMs to be conditionally
        independent. Therefore, **this function returns the corresponding
        conditionally independent state-by-state TPM.**

    .. note::
        The indices of the rows of the state-by-node TPM are assumed to follow
        the little-endian convention, while the indices of the columns follow
        the big-endian convention. The indices of the rows and columns of the
        resulting state-by-state TPM both follow the big-endian convention. See
        the documentation on PyPhi :ref:`tpm-conventions` for more info.

    Args:
        tpm (list[list] or np.ndarray): A state-by-node TPM with row indices
            following the little-endian convention and column indices following
            the big-endian convention.

    Returns:
        np.ndarray: A state-by-state TPM, with both row and column indices
        following the big-endian convention.

    Examples:
    >>> tpm = np.array([[1, 1, 0],
    ...                 [0, 0, 1],
    ...                 [0, 1, 1],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [1, 1, 1],
    ...                 [1, 0, 1]])
    >>> state_by_node2state_by_state(tpm)
    array([[0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1., 0., 0.]])
    >>> tpm = np.array([[0.1, 0.3, 0.7],
    ...                 [0.3, 0.9, 0.2],
    ...                 [0.3, 0.9, 0.1],
    ...                 [0.2, 0.8, 0.5],
    ...                 [0.1, 0.7, 0.4],
    ...                 [0.4, 0.3, 0.6],
    ...                 [0.4, 0.3, 0.1],
    ...                 [0.5, 0.2, 0.1]])
    >>> state_by_node2state_by_state(tpm)
    array([[0.189, 0.021, 0.081, 0.009, 0.441, 0.049, 0.189, 0.021],
           [0.056, 0.024, 0.504, 0.216, 0.014, 0.006, 0.126, 0.054],
           [0.063, 0.027, 0.567, 0.243, 0.007, 0.003, 0.063, 0.027],
           [0.08 , 0.02 , 0.32 , 0.08 , 0.08 , 0.02 , 0.32 , 0.08 ],
           [0.162, 0.018, 0.378, 0.042, 0.108, 0.012, 0.252, 0.028],
           [0.168, 0.112, 0.072, 0.048, 0.252, 0.168, 0.108, 0.072],
           [0.378, 0.252, 0.162, 0.108, 0.042, 0.028, 0.018, 0.012],
           [0.36 , 0.36 , 0.09 , 0.09 , 0.04 , 0.04 , 0.01 , 0.01 ]])
    """
    # Reshape to 2D
    sbn_tpm = to_2dimensional(tpm)
    # Get number of previous states
    Sp = sbn_tpm.shape[0]
    # Get number of nodes in the next state
    Nn = sbn_tpm.shape[1]
    # Get the number of next states
    Sn = 2**Nn
    if is_deterministic(tpm):
        return _deterministic_sbn2sbs(Sp, Sn, sbn_tpm)
    else:
        return _nondeterministic_sbn2sbs(Nn, Sn, sbn_tpm)


# Short aliases

b2l = be2le
l2b = le2be
l2s = le_index2state
b2s = be_index2state
s2l = state2le_index
s2b = state2be_index
b2l_sbs = be2le_state_by_state
l2b_sbs = le2be_state_by_state
to_md = to_multidimensional
to_2d = to_2dimensional
sbn2sbs = state_by_node2state_by_state
sbs2sbn = state_by_state2state_by_node
