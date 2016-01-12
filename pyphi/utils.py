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
from itertools import chain, combinations, product
from scipy.misc import comb
from scipy.spatial.distance import cdist
from pyemd import emd
from .cache import cache
from . import constants

# Create a logger for this module.
log = logging.getLogger(__name__)


# TPM and Connectivity Matrix utils
# ============================================================================

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


def fully_connected(connectivity_matrix, nodes1, nodes2):
    """Tests connectivity of one set of nodes to another.

    Args:
        connectivity_matrix (``np.ndarrray``): The connectivity matrix
        nodes1 (tuple(int)): The nodes whose outputs to ``nodes2`` will be
            tested.
        nodes2 (tuple(int)): The nodes whose inputs from ``nodes1`` will
            be tested.

    Returns:
        bool: Returns True if all elements in ``nodes1`` output to
            some element in ``nodes2`` AND all elements in ``nodes2``
            have an input from some element in ``nodes1``. Otherwise
            return False. Return True if either set of nodes is empty.
    """
    if not nodes1 or not nodes2:
        return True

    cm = submatrix(connectivity_matrix, nodes1, nodes2)

    # Do all nodes have at least one connection?
    return cm.sum(0).all() and cm.sum(1).all()


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


def get_inputs_from_cm(index, connectivity_matrix):
    """Returns a tuple of node indices that have connections to the node with
    the given index."""
    return tuple(i for i in range(connectivity_matrix.shape[0]) if
                 connectivity_matrix[i][index])


def get_outputs_from_cm(index, connectivity_matrix):
    """Returns a tuple of node indices that the node with the given index has
    connections to."""
    return tuple(i for i in range(connectivity_matrix.shape[0]) if
                 connectivity_matrix[index][i])


def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)


def phi_eq(x, y):
    """Compare two phi values up to |PRECISION|."""
    return abs(x - y) <= constants.EPSILON


# see http://stackoverflow.com/questions/16003217
def combs(a, r):
    """NumPy implementation of itertools.combinations.

    Return successive |r|-length combinations of elements in the array ``a``.

    Args:
      a (np.ndarray): The array from which to get combinations.
      r (int): The length of the combinations.

    Returns:
        combinations (``np.ndarray``): An array of combinations.
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
        combination_indices (``np.ndarray``): Indices that give the
            |k|-combinations of |n| elements.

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
        chain (``Iterable``): An chained iterator over the power set.

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
        distribution (``np.ndarray``): The uniform distribution over the set of
            nodes.
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
        tpm (``np.ndarray``): A TPM with the same number of dimensions, with
            the node marginalized out.
    """
    if perturb_value == 0.5:
        return tpm.sum(index, keepdims=True) / tpm.shape[index]
    else:
        tpm = np.average(tpm, index, weights=[1 - perturb_value, perturb_value])
        return tpm.reshape([i for i in tpm.shape[0:index]] +
                           [1] + [i for i in tpm.shape[index:]])


@cache(cache={}, maxmem=None)
def max_entropy_distribution(node_indices, number_of_nodes,
                             perturb_vector=None):
    """Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside ``node_indices`` are fixed and treated as if they have only 1
    state.

    Args:
        node_indices (tuple(int)): The set of node indices over which to take
            the distribution.
        number_of_nodes (int): The total number of nodes in the network.

    Returns:
        distribution (``np.ndarray``): The maximum entropy distribution over
            the set of nodes.
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
    # transportation cost function.
    return emd(d1.ravel(), d2.ravel(), _hamming_matrix(d1.ndim))


def bipartition(a):
    """ Return a list of bipartitions for a sequence.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        bipartition (``list(tuple(tuple))``): A list of tuples containing each
            of the two partitions.

    Example:
        >>> from pyphi.utils import bipartition
        >>> bipartition((1,2,3))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,))]
    """
    return [(tuple(a[i] for i in part0_idx), tuple(a[j] for j in part1_idx))
            for part0_idx, part1_idx in bipartition_indices(len(a))]


# TODO? [optimization] optimize this to use indices rather than nodes
# TODO? are native lists really slower
def directed_bipartition(a):
    """Return a list of directed bipartitions for a sequence.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        bipartition (``list(tuple(tuple))``): A list of tuples containing each
            of the two partitions.

    Example:
        >>> from pyphi.utils import directed_bipartition
        >>> directed_bipartition((1, 2, 3))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)), ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    """
    return [(tuple(a[i] for i in part0_idx), tuple(a[j] for j in part1_idx))
            for part0_idx, part1_idx in directed_bipartition_indices(len(a))]


def directed_bipartition_of_one(a):
    """Return a list of directed bipartitions for a sequence where each
    bipartitions includes a set of size 1.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        bipartition (``list(tuple(tuple))``): A list of tuples containing each
            of the two partitions.

    Example:
        >>> from pyphi.utils import directed_bipartition_of_one
        >>> directed_bipartition_of_one((1,2,3))
        [((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)), ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,))]
    """
    return [partition for partition in directed_bipartition(a)
            if len(partition[0]) == 1 or len(partition[1]) == 1]


@cache(cache={}, maxmem=None)
def directed_bipartition_indices(N):
    """Returns indices for directed bipartitions of a sequence.

    The directed bipartion

    Args:
        N (int): The length of the sequence.

    Returns:
        bipartition_indices (``list``): A list of tuples containing the indices
            for each of the two partitions.

    Example:
        >>> from pyphi.utils import directed_bipartition_indices
        >>> N = 3
        >>> directed_bipartition_indices(N)
        [((), (0, 1, 2)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 1), (2,)), ((2,), (0, 1)), ((0, 2), (1,)), ((1, 2), (0,)), ((0, 1, 2), ())]
    """
    indices = bipartition_indices(N)
    return indices + [idx[::-1] for idx in indices[::-1]]


@cache(cache={}, maxmem=None)
def bipartition_indices(N):
    """Returns indices for bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        bipartition_indices (``list``): A list of tuples containing the indices
            for each of the two partitions.

    Example:
        >>> from pyphi.utils import bipartition_indices
        >>> N = 3
        >>> bipartition_indices(N)
        [((), (0, 1, 2)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 1), (2,))]
    """
    result = []
    if N <= 0:
        return result

    for i in range(2 ** (N-1)):
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
    for i in range(_NUM_PRECOMPUTED_HAMMING_MATRICES)
]


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        hamming_matrix (``np.ndarray``): A |2^N x 2^N| matrix where the |ith|
            element is the Hamming distance between state |i| and state |j|.

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


def submatrix(cm, nodes1, nodes2):
    """Return the submatrix of connections from ``nodes1`` to ``nodes2``.

    Args:
        cm (np.ndarray): The matrix
        nodes1 (tuple(int)): Source nodes
        nodes2 (tuple(int)): Sink nodes
    """
    submatrix_indices = np.ix_(nodes1, nodes2)
    return cm[submatrix_indices]


#TODO: better name?
def relevant_connections(n, _from, to):
    """Construct a connectivity matrix.

    Returns an |n x n| connectivity matrix with the |i,jth| entry
    set to ``1`` if |i| is in ``_from`` and |j| is in ``to``.

    Args:
        n (int): The dimensions of the matrix
        _from (tuple(int)): Nodes with outgoing connections to ``to``
        to (tuple(int)): Nodes with incoming connections from ``_from``
    """
    cm = np.zeros((n, n))

    # Don't try and index with empty arrays. Older versions of NumPy
    # (at least up to 1.9.3) break with empty array indices.
    if not _from or not to:
        return cm

    cm[np.ix_(_from, to)] = 1
    return cm


def block_cm(cm):
    """Return whether ``cm`` can be arranged as a block connectivity matrix.

    If so, the corresponding mechanism/purview is trivially reducible.
    Technically, only square matrices are "block diagonal", but the notion of
    connectivity carries over.

    We test for block connectivity by trying to grow a block of nodes such
    that:

    * 'source' nodes only input to nodes in the block
    * 'sink' nodes only receive inputs from source nodes in the block

    For example, the following connectivity matrix represents connections from
    ``nodes1 = A, B, C`` to ``nodes2 = D, E, F, G`` (without loss of
    generality—note that ``nodes1`` and ``nodes2`` may share elements)::

         D  E  F  G
      A [1, 1, 0, 0]
      B [1, 1, 0, 0]
      C [0, 0, 1, 1]

    Since nodes |AB| only connect to nodes |DE|, and node |C| only connects to
    nodes |FG|, the subgraph is reducible; the cut ::

      AB   C
      -- X --
      DE   FG

    does not change the structure of the graph.
    """
    if np.any(cm.sum(1) == 0):
        return True
    if np.all(cm.sum(1) == 1):
        return True

    outputs = list(range(cm.shape[1]))

    # CM helpers:
    # All nodes that `nodes` connect (output) to
    outputs_of = lambda nodes: np.where(cm[nodes, :].sum(0))[0]
    # All nodes which connect (input) to `nodes`
    inputs_to = lambda nodes: np.where(cm[:, nodes].sum(1))[0]

    # Start: source node with most outputs
    sources = [np.argmax(cm.sum(1))]
    sinks = outputs_of(sources)
    sink_inputs = inputs_to(sinks)

    while True:
        if np.all(sink_inputs == sources):
            # sources exclusively connect to sinks.
            # There are no other nodes which connect sink nodes,
            # hence set(sources) + set(sinks) form a component
            # which is not connected to the rest of the graph
            return True

        # Recompute sources, sinks, and sink_inputs
        sources = sink_inputs
        sinks = outputs_of(sources)
        sink_inputs = inputs_to(sinks)

        # Considering all output nodes?
        if np.all(sinks == outputs):
            return False


# TODO: simplify the conditional validation here and in block_cm
# TODO: combine with fully_connected
def block_reducible(cm, nodes1, nodes2):
    """Return whether connections from ``nodes1`` to ``nodes2`` are reducible.

    Args:
        cm (np.ndarray): The network's connectivity matrix.
        nodes1 (tuple(int)): Source nodes
        nodes2 (tuple(int)): Sink nodes
    """
    if not nodes1 or not nodes2:
        return True  # trivially

    cm = submatrix(cm, nodes1, nodes2)

    # Validate the connectivity matrix.
    if not cm.sum(0).all() or not cm.sum(1).all():
        return True
    if len(nodes1) > 1 and len(nodes2) > 1:
        return block_cm(cm)
    return False


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
