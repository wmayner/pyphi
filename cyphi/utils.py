#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities
~~~~~~~~~

This module provides utility functions used within CyPhi that are consumed
by more than one class.
"""

import numpy as np
from re import match
from collections import namedtuple
from itertools import chain, combinations
from copy import copy
from scipy.misc import comb
from scipy.spatial.distance import cdist
from pyemd import emd as _emd


def cut(subsystem, cut):
    """Returns a copy of the subsystem with given cut applied."""
    cut_subsystem = copy(subsystem)
    cut_subsystem.cut(cut)
    return cut_subsystem


def tuple_eq(a, b):
    """Return whether two tuples are equal, using ``np.array_equal`` for
    numpy arrays.
    """
    # Use numpy equality if both are numpy arrays
    if isinstance(a, type(np.array([]))) and isinstance(b, type(np.array([]))):
        return np.array_equal(a, b)

    try:
        # Shortcircuit if arguments are difference lengths
        if len(a) != len(b):
            return False
    # Fall back to normal equality if we have a non-iterable argument
    except TypeError:
        return a == b

    # Otherwise iterate through and try normal equality, recursing if that
    # fails
    result = True
    for i in range(len(a)):
        try:
            if not a[i] == b[i]:
                return False
        except ValueError as e:
            if (str(e) == "The truth value of an array with more than one " +
                          "element is ambiguous. Use a.any() or a.all()"):
                return tuple_eq(a[i], b[i])
            else:
                raise e
    return result


# see http://stackoverflow.com/questions/16003217
def combs(a, r):
    """
    NumPy implementation of itertools.combinations.

    Return successive |r|-length combinations of elements in the array ``a``.

    :param a: the array from which to get combinations
    :type a: ``np.ndarray``
    :param r:  the length of the combinations
    :type r: ``int``

    :returns: An array of combinations
    :rtype: ``np.ndarray``
    """
    # Special-case for 0-length combinations
    if r is 0:
        return np.asarray([])

    a = np.asarray(a)
    data_type = a.dtype if r is 0 else np.dtype([('', a.dtype)] * r)
    b = np.fromiter(combinations(a, r), data_type)
    return b.view(a.dtype).reshape(-1, r)


# see http://stackoverflow.com/questions/16003217/
def comb_indices(n, k):
    """
    N-D version of itertools.combinations.

    Return indices that yeild the |r|-combinations of |n| elements.

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

    :param a: array from which to get combinations
    :type a: ``np.ndarray``
    :param k: length of combinations
    :type k: ``int``

    :returns: Indices of the |r|-combinations of |n| elements
    :rtype: ``np.ndarray``
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
    """
    Return the power set of an iterable (see `itertools recipes
    <http://docs.python.org/2/library/itertools.html#recipes>`_).

        >>> ps = powerset(np.arange(2))
        >>> print(list(ps))
        [(), (0,), (1,), (0, 1)]

    :param iterable: The iterable from which to generate the power set
    :type iterable: iterable

    :returns: An iterator over the power set
    :rtype: iterator
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def uniform_distribution(number_of_nodes):
    """
    Return the uniform distribution for a set of binary nodes, indexed by state
    (so there is one dimension per node, the size of which is the number of
    possible states for that node).

    :param nodes: a set of indices of binary nodes
    :type nodes: ``np.ndarray``

    :returns: The uniform distribution over the set of nodes
    :rtype: ``np.ndarray``
    """
    # The size of the state space for binary nodes is 2^(number of nodes).
    number_of_states = 2 ** number_of_nodes
    # Generate the maximum entropy distribution
    # TODO extend to nonbinary nodes
    return np.divide(np.ones(number_of_states),
                     number_of_states).reshape([2] * number_of_nodes)


def marginalize_out(node, tpm):
    """
    Marginalize out a node from a TPM.

    The TPM must be indexed by individual node state.

    :param node: The node to be marginalized out
    :type node: ``Node``
    :param tpm: The tpm to marginalize the node out of
    :type tpm: ``np.ndarray``

    :returns: The TPM after marginalizing out the node
    :rtype: ``np.ndarray``
    """
    # Preserve number of dimensions so node indices still index into the proper
    # axis of the returned distribution, normalize the distribution by number
    # of states
    return np.divide(np.sum(tpm, node.index, keepdims=True),
                     tpm.shape[node.index])


# TODO memoize this
def max_entropy_distribution(nodes, network):
    """
    Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside the are fixed and treated as if they have only 1 state.

    :param nodes: The set of nodes
    :type nodes: ``[Node]``
    :param network: The network the nodes belong to
    :type network: ``Network``

    :returns: The maximum entropy distribution over this subsystem
    :rtype: ``np.ndarray``
    """
    # TODO extend to nonbinary nodes
    max_ent_shape = [2 if node in nodes else 1 for node in network.nodes]
    return np.divide(np.ones(max_ent_shape),
                     np.ufunc.reduce(np.multiply, max_ent_shape))


# TODO extend to binary nodes
# TODO? parametrize and use other metrics (KDL, L1)
# TODO ensure that we really don't need to keep track of the states for the
#      correct hamming distance... are we really only comparing the same
#      purviews?
def emd(d1, d2):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node).

    Singleton dimensions are sqeezed out.
    """
    if d1.shape != d2.shape:
        raise ValueError("Distributions must be the same shape.")
    d1, d2 = d1.squeeze(), d2.squeeze()
    # Compute the EMD with Hamming distance between states as the
    # transportation cost function
    return _emd(d1.ravel(), d2.ravel(), _hamming_matrix(d1.ndim))


# TODO? [optimization] optimize this to use indices rather than nodes
# TODO? are native lists really slower
def bipartition(a):
    """Generates all bipartitions for a sequence or ``np.array``.

        >>> from cyphi.utils import bipartition
        >>> list(bipartition([1, 2, 3]))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,))]

    :param array: The list to partition
    :type array: ``[], (), or np.ndarray``

    :returns: A generator that yields a tuple containing each of the two
        partitions (lists of nodes)
    :rtype: ``generator``
    """
    # Get size of list or array and validate
    if isinstance(a, type(np.array([]))):
        size = a.size
    else:
        size = len(a)
    # Return on empty input
    if size <= 0:
        return

    for bitstring in [bin(i)[2:].zfill(size)[::-1]
                      for i in range(2 ** (size - 1))]:
        yield (_bitstring_index(a, bitstring),
               _bitstring_index(a, _flip(bitstring)))


# Internal helper methods
# =============================================================================


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

        >>> from cyphi.utils import _hamming_matrix
        >>> _hamming_matrix(2)
        array([[ 0.,  1.,  1.,  2.],
               [ 1.,  0.,  2.,  1.],
               [ 1.,  2.,  0.,  1.],
               [ 2.,  1.,  1.,  0.]])

    :param N: The number of nodes under consideration
    :type N: ``int``

    :returns: A |2^N x 2^N| matrix where the |ith| element is the Hamming
        distance between state |i| and state |j|.
    :rtype: ``np.ndarray``
    """
    possible_states = np.array([list(bin(state)[2:].zfill(N)) for state in
                                range(2 ** N)])
    return cdist(possible_states, possible_states, 'hamming') * N


def _bitstring_index(a, bitstring):
    """Select elements of a sequence or ``np.array`` based on a binary string.

    The |ith| element in the array is selected if there is a 1 at the |ith|
    position of the bitstring.

        >>> from cyphi.utils import _bitstring_index
        >>> bitstring = '10010100'
        >>> _bitstring_index([0, 1, 2, 3, 4, 5, 6, 7], bitstring)
        (0, 3, 5)

    :param a: The sequence or ``np.array`` to select from.
    :type a: ``sequence or np.ndarray``
    :param bitstring: The binary string indicating which elements are to be
        selected.
    :type bitstring: ``str``

    :returns: A list of all the elements at indices where there is a 1 in the
        binary string
    :rtype: ``tuple``
    """
    # Get size of iterable and validate
    if isinstance(a, type(np.array([]))):
        if a.ndim != 1:
            raise ValueError("Array must be 1-dimensional.")
        size = a.size
    else:
        size = len(a)

    if size != len(bitstring):
        raise ValueError("The bitstring must be the same length as the array.")
    if not match('^[01]*$', bitstring):
        raise ValueError("Bitstring must contain only 1s and 0s. Did you " +
                         "forget to chop off the first two characters after " +
                         "using `bin`?")

    return tuple(a[i] for i in range(size) if bitstring[i] == '1')


def _flip(bitstring):
    """Flip the bits in a string consisting of 1s and zeros."""
    return ''.join('1' if x == '0' else '0' for x in bitstring)


# TODO? implement this
def connectivity_matrix_to_tpm(connectivity_matrix):
    """
    :param connectivity_matrix: The network's connectivity matrix (must be
        square)
    :type connectivity_matrix: ``np.ndarray``
    :param tpm: The network's transition probability matrix (state-by-node
        form)
    :type tpm: ``np.ndarray``
    """
    # Ensure connectivity matrix is square
    if ((len(connectivity_matrix.shape) is not 2) or
            (connectivity_matrix.shape[0] is not
                connectivity_matrix.shape[1])):
        raise ValueError("Connectivity matrix must be square.")


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
