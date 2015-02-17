#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# utils.py
"""
Functions used by more than one PyPhi module or class, or that might be of
external use.
"""

import re
import logging
import hashlib
import numpy as np
from itertools import chain, combinations
from scipy.misc import comb
from scipy.spatial.distance import cdist
from pyemd import emd
from .lru_cache import lru_cache
from . import constants, config


# Create a logger for this module.
log = logging.getLogger(__name__)


def condition_tpm(tpm, fixed_nodes, state):
    """Return a TPM conditioned on the given fixed node indices, whose states
    are fixed according to the given state-tuple.

    The dimensions of the new TPM that correspond to the fixed nodes are
    collapsed onto their state, making those dimensions singletons suitable for
    broadcasting. The number of dimensions of the conditioned TPM will be the
    same as the unconditioned TPM."""
    conditioning_indices = [[slice(None)]] * len(state)
    for i in fixed_nodes:
        # Preserve singleton dimensions with `np.newaxis`
        conditioning_indices[i] = [state[i], np.newaxis]
    # Flatten the indices.
    conditioning_indices = list(chain.from_iterable(conditioning_indices))
    # Obtain the actual conditioned TPM by indexing with the conditioning
    # indices.
    return tpm[conditioning_indices]


# TODO test
def apply_cut(cut, connectivity_matrix):
    """Returns a modified connectivity matrix where the connections from one
    set of nodes to the other are destroyed."""
    if cut is None:
        return connectivity_matrix
    cm = connectivity_matrix.copy()
    for i in cut.severed:
        for j in cut.intact:
            cm[i][j] = 0
    return cm


def apply_boundary_conditions_to_cm(external_indices, connectivity_matrix):
    """Returns a connectivity matrix with all connections to or from external
    nodes removed."""
    cm = connectivity_matrix.copy()
    for i in external_indices:
        # Zero-out row
        cm[i] = 0
        # Zero-out column
        cm[:, i] = 0
    return cm


# TODO test
def get_inputs_from_cm(index, connectivity_matrix):
    """Returns a tuple of node indices that have connections to the node with
    the given index."""
    return tuple(i for i in range(connectivity_matrix.shape[0]) if
                 connectivity_matrix[i][index])


# TODO test
def get_outputs_from_cm(index, connectivity_matrix):
    """Returns a tuple of node indices that the node with the given index has
    connections to."""
    return tuple(i for i in range(connectivity_matrix.shape[0]) if
                 connectivity_matrix[index][i])


def np_hash(a):
    """Return a hash of a NumPy array.

    This is much faster than ``np.toString`` for large arrays."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)


def phi_eq(x, y):
    """Compare two phi values up to |PRECISION|."""
    return abs(x - y) < constants.EPSILON


# see http://stackoverflow.com/questions/16003217
def combs(a, r):
    """NumPy implementation of itertools.combinations.

    Return successive |r|-length combinations of elements in the array ``a``.

    Args:
      a (np.ndarray): The array from which to get combinations.
      r (int): The length of the combinations.

    Returns:
        ``np.ndarray`` -- An array of combinations.
    """
    # Special-case for 0-length combinations
    if r == 0:
        return np.asarray([])

    a = np.asarray(a)
    data_type = a.dtype if r == 0 else np.dtype([('', a.dtype)] * r)
    b = np.fromiter(combinations(a, r), data_type)
    return b.view(a.dtype).reshape(-1, r)


# see http://stackoverflow.com/questions/16003217/
def comb_indices(n, k):
    """N-D version of itertools.combinations.

    Args:
        a (np.ndarray): The array from which to get combinations.
        k (int): The desired length of the combinations.

    Returns:
        ``np.ndarray`` -- Indices that give the |k|-combinations of |n|
        elements.

    Example:
        >>> n, k = 3, 2
        >>> data = np.arange(6).reshape(2, 3)
        >>> data[:, comb_indices(n, k)]
        array([[[0, 1],
                [0, 2],
                [1, 2]],
        <BLANKLINE>
               [[3, 4],
                [3, 5],
                [4, 5]]])
    """
    # Count the number of combinations for preallocation
    count = comb(n, k, exact=True)
    # Get numpy iterable from ``itertools.combinations``
    indices = np.fromiter(
        chain.from_iterable(combinations(range(n), k)),
        int,
        count=(count * k))
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


# TODO? implement this with numpy
def powerset(iterable):
    """Return the power set of an iterable (see `itertools recipes
    <http://docs.python.org/2/library/itertools.html#recipes>`_).

    Args:
        iterable (Iterable): The iterable from which to generate the power set.

    Returns:
        ``chain`` -- An chained iterator over the power set.

    Example:
        >>> ps = powerset(np.arange(2))
        >>> print(list(ps))
        [(), (0,), (1,), (0, 1)]
    """
    return chain.from_iterable(combinations(iterable, r)
                               for r in range(len(iterable) + 1))


def uniform_distribution(number_of_nodes):
    """
    Return the uniform distribution for a set of binary nodes, indexed by state
    (so there is one dimension per node, the size of which is the number of
    possible states for that node).

    Args:
        nodes (np.ndarray): A set of indices of binary nodes.

    Returns:
        ``np.ndarray`` -- The uniform distribution over the set of nodes.
    """
    # The size of the state space for binary nodes is 2^(number of nodes).
    number_of_states = 2 ** number_of_nodes
    # Generate the maximum entropy distribution
    # TODO extend to nonbinary nodes
    return (np.ones(number_of_states) /
            number_of_states).reshape([2] * number_of_nodes)


def marginalize_out(index, tpm, perturb_value=0.5):
    """
    Marginalize out a node from a TPM.

    Args:
        index (list): The index of the node to be marginalized out.
        tpm (np.ndarray): The TPM to marginalize the node out of.

    Returns:
        ``np.ndarray`` -- A TPM with the same number of dimensions, with the
        node marginalized out.
    """
    if (perturb_value == 0.5):
        return tpm.sum(index, keepdims=True) / tpm.shape[index]
    else:
        tpm = np.average(tpm, index, weights=[1 - perturb_value, perturb_value])
        return tpm.reshape([i for i in tpm.shape[0:index]] +
                           [1] + [i for i in tpm.shape[index:]])


@lru_cache(cache={}, maxmem=config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)
def max_entropy_distribution(node_indices, number_of_nodes,
                             perturb_vector=None):
    """
    Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside the are fixed and treated as if they have only 1 state.

    Args:
        nodes (tuple(Nodes)): The set of nodes over which to take the
            distribution.
        network (Network): The network the nodes belong to.

    Returns:
        ``np.ndarray`` -- The maximum entropy distribution over the set of
        nodes.
    """
    # TODO extend to nonbinary nodes
    if ((perturb_vector is None) or
            (np.all(perturb_vector == 0.5)) or
            (len(perturb_vector) == 0)):
        distribution = np.ones([2 if index in node_indices else 1 for index in
                                range(number_of_nodes)])
        return distribution / distribution.size
    else:
        perturb_vector = np.array(perturb_vector)
        bin_states = [bin(x)[2:].zfill(len(node_indices))[::-1] for x in
                      range(2 ** len(node_indices))]
        distribution = np.array([
            np.prod(perturb_vector[[m.start() for m in
                                    re.finditer('1', bin_states[x])]])
            * np.prod(1 - perturb_vector[[m.start() for m in
                                          re.finditer('0', bin_states[x])]])
            for x in range(2 ** len(node_indices))
        ])
        return distribution.reshape(
            [2 if index in node_indices else 1 for index in
             range(number_of_nodes)],
            order='F')


# TODO extend to binary nodes
# TODO? parametrize and use other metrics (KDL, L1)
def hamming_emd(d1, d2):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node).

    Singleton dimensions are sqeezed out.
    """
    d1, d2 = d1.squeeze(), d2.squeeze()
    # Compute the EMD with Hamming distance between states as the
    # transportation cost function
    return emd(d1.ravel(), d2.ravel(), _hamming_matrix(d1.ndim))


# TODO? [optimization] optimize this to use indices rather than nodes
# TODO? are native lists really slower
def bipartition(a):
    """Return a list of bipartitions for a sequence.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        ``list`` -- A list of tuples containing each of the two partitions.

    Example:
        >>> from pyphi.utils import bipartition
        >>> bipartition((1, 2, 3))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,))]
    """
    return [(tuple(a[i] for i in part0_idx), tuple(a[j] for j in part1_idx))
            for part0_idx, part1_idx in bipartition_indices(len(a))]


@lru_cache(cache={}, maxmem=config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)
def bipartition_indices(N):
    """Returns indices for bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        ``list`` -- A list of tuples containing the indices for each of the two
        partitions.

    Example:
        >>> from pyphi.utils import bipartition_indices
        >>> N = 3
        >>> bipartition_indices(N)
        [((), (0, 1, 2)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 1), (2,))]
    """
    result = []
    # Return on empty input
    if N <= 0:
        return result
    for i in range(2 ** (N - 1)):
        part = [[], []]
        for n in range(N):
            bit = (i >> n) & 1
            part[bit].append(n)
        result.append((tuple(part[1]), tuple(part[0])))
    return result


# Internal helper methods
# =============================================================================


# Load precomputed hamming matrices.
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = [
    np.load(os.path.join(_ROOT, 'data', 'hamming_matrices', str(i) + '.npy'))
    for i in range( _NUM_PRECOMPUTED_HAMMING_MATRICES)
]
# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        ``np.ndarray`` -- A |2^N x 2^N| matrix where the |ith| element is the
        Hamming distance between state |i| and state |j|.

    Example:
        >>> from pyphi.utils import _hamming_matrix
        >>> _hamming_matrix(2)
        array([[ 0.,  1.,  1.,  2.],
               [ 1.,  0.,  2.,  1.],
               [ 1.,  2.,  0.,  1.],
               [ 2.,  1.,  1.,  0.]])
    """
    if N < 10:
        return _hamming_matrices[N]
    else:
        log.warn(
            "Hamming matrices for more than {} nodes have not been "
            "precomputed. This will make EMD calculations less inefficient; "
            "calculating hamming matrices is an exponential-time procedure. "
            "Consider pre-computing the hamming matrices up to the desired "
            "number of nodes with the ``pyphi.utils._hamming_matrix`` "
            "function and saving them to the 'data' directory in the "
            "directory where PyPhi was installed (you can find this directory "
            "by typing ``import pyphi; pyphi;`` into a Python interperter)."
            .format(_NUM_PRECOMPUTED_HAMMING_MATRICES - 1)
        )
        possible_states = np.array([list(bin(state)[2:].zfill(N)) for state in
                                    range(2 ** N)])
        return cdist(possible_states, possible_states, 'hamming') * N


# TODO? implement this
def connectivity_matrix_to_tpm(network):
    """Generate a TPM from a connectivity matrix and nodes that implement
    logical functions.

    Args:
        network (Network): The network for which to generate the TPM.

    Returns:
        ``np.ndarray`` -- A transition probability matrix.
    """


# Custom printing methods
# =============================================================================


def print_repertoire(r):
    print('\n', '-' * 80)
    for i in range(r.size):
        strindex = bin(i)[2:].zfill(r.ndim)
        index = tuple(map(int, list(strindex)))
        print('\n', strindex, '\t', r[index])
    print('\n', '-' * 80, '\n')


def print_repertoire_horiz(r):
    r = np.squeeze(r)
    colwidth = 11
    print('\n' + '-' * 70 + '\n')
    index_labels = [bin(i)[2:].zfill(r.ndim) for i in range(r.size)]
    indices = [tuple(map(int, list(s))) for s in index_labels]
    print('     p:  ', '|'.join('{0:.3f}'.format(r[index]).center(colwidth) for
                                index in indices))
    print('         ', '|'.join(' ' * colwidth for index in indices))
    print(' state:  ', '|'.join(label.center(colwidth) for label in
                                index_labels))
    print('\n' + '-' * 70 + '\n')


def print_partition(p):
    print('\nPart 1: \n\n', p[0].mechanism, '\n-----------------\n',
          p[0].purview)
    print('\nPart 2: \n\n', p[1].mechanism, '\n-----------------\n',
          p[1].purview, '\n')
