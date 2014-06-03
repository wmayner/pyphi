#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matlab
~~~~~~

Functions to aid conversion from data structures used in the Matlab IIT code to
their CyPhi equivalents.
"""

import numpy as np


def state2matlab_index(state):
    """Convert a CyPhi state tuple to a decimal integer for indexing into a
    Matlab-style TPM.

    Examples:
        >>> from cyphi.matlab import state2matlab_index
        >>> state2matlab_index((1, 0, 0, 0, 0))
        1
        >>> state2matlab_index((1, 1, 1, 0, 0))
        7
    """
    return int(''.join(map(str, state[::-1])), 2)


def cyphi_index2matlab_index(i, number_of_nodes):
    """Converts a decimal integer index for a Cyphi TPM to that of a
    Matlab TPM.

    Examples:
        >>> from cyphi.matlab import cyphi_index2matlab_index
        >>> number_of_nodes = 3
        >>> cyphi_index2matlab_index(4, number_of_nodes)
        1
        >>> cyphi_index2matlab_index(6, number_of_nodes)
        3
    """
    return int(bin(i)[2:].zfill(number_of_nodes)[::-1], 2)


def matlab_tpm2cyphi_tpm(matlab_tpm):
    """Convert a 2-D TPM from the Matlab IIT code to a CyPhi-style 2-D TPM.

    In 2-D CyPhi TPMs, row |i| gives the transition probabilities for the state
    given by the binary representation of |i|.

    In Matlab TPMs, row |i| gives the transition probabilities for the state
    given by the **reverse** of the binary representation of |i| (so that, for
    example, the decimal integer 1 corresponds to the first node being on and
    all others being off).

    Example:
        >>> from cyphi.matlab import matlab_tpm2cyphi_tpm
        >>> matlab_tpm = np.array(
        ... [[0, 0, 0],
        ...  [0, 0, 1],
        ...  [1, 0, 1],
        ...  [1, 0, 0],
        ...  [1, 0, 0],
        ...  [1, 1, 1],
        ...  [1, 0, 1],
        ...  [1, 1, 0]])
        >>> matlab_tpm2cyphi_tpm(matlab_tpm)
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 1.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  0.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  0.,  0.],
               [ 1.,  1.,  0.]])
    """
    number_of_states = matlab_tpm.shape[0]
    number_of_nodes = matlab_tpm.shape[1]
    # Preallocate the CyPhi TPM
    # TODO extend to nonbinary nodes
    cyphi_tpm = np.zeros([2 ** number_of_nodes, number_of_nodes])
    for i in range(number_of_states):
        cyphi_tpm[i] = matlab_tpm[cyphi_index2matlab_index(i, number_of_nodes)]
    return cyphi_tpm
