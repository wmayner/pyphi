#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/distance.py

"""
Functions for measuring distances.
"""

import numpy as np
from pyemd import emd
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from . import constants, utils
from .distribution import flatten

# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = utils.load_data('hamming_matrices',
                                    _NUM_PRECOMPUTED_HAMMING_MATRICES)


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        np.ndarray: A |2^N x 2^N| matrix where the |ith| element is the Hamming
        distance between state |i| and state |j|.

    Example:
        >>> _hamming_matrix(2)
        array([[ 0.,  1.,  1.,  2.],
               [ 1.,  0.,  2.,  1.],
               [ 1.,  2.,  0.,  1.],
               [ 2.,  1.,  1.,  0.]])
    """
    if N < _NUM_PRECOMPUTED_HAMMING_MATRICES:
        return _hamming_matrices[N]
    return _compute_hamming_matrix(N)


@constants.joblib_memory.cache
def _compute_hamming_matrix(N):
    """
    Compute and store a Hamming matrix for |N| nodes.

    Hamming matrices have the following sizes:

    n   MBs
    ==  ===
    9   2
    10  8
    11  32
    12  128
    13  512

    Given these sizes and the fact that large matrices are needed infrequently,
    we store computed matrices using the Joblib filesystem cache instead of
    adding computed matrices to the ``_hamming_matrices`` global and clogging
    up memory.

    This function is only called when N > _NUM_PRECOMPUTED_HAMMING_MATRICES.
    Don't call this function directly; use :func:`_hamming_matrix` instead.
    """
    possible_states = np.array(list(utils.all_states((N))))
    return cdist(possible_states, possible_states, 'hamming') * N


# TODO extend to binary nodes
def hamming_emd(d1, d2):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node) using the Hamming distance between states
    as the transportation cost function.

    Singleton dimensions are sqeezed out.
    """
    N = d1.squeeze().ndim
    d1, d2 = flatten(d1), flatten(d2)
    return emd(d1, d2, _hamming_matrix(N))


def l1(d1, d2):
    """Return the L1 distance between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The sum of absolute differences of ``d1`` and ``d2``.
    """
    return np.absolute(d1 - d2).sum()


def kld(d1, d2):
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The KLD of ``d1`` from ``d2``.
    """
    d1, d2 = flatten(d1), flatten(d2)
    return entropy(d1, d2, 2.0)


def entropy_difference(d1, d2):
    """Return the difference in entropy between two distributions."""
    d1, d2 = flatten(d1), flatten(d2)
    return abs(entropy(d1, base=2.0) - entropy(d2, base=2.0))
