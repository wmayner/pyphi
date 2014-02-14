# -*- coding: utf-8 -*-

"""
cyphi.utils
~~~~~~~~~~~

This module provides utility functions used within CyPhi that are also useful
for external consumption.

"""

import numpy as np
from itertools import chain, combinations
from scipy.misc import comb
from .exceptions import ValidationException


# see http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
def combs(a, r):
    """
    NumPy implementation of itertools.combinations.

    Return successive :math:`r`-length combinations of elements in the array
    `a`.

    :param a: the array from which to get combinations
    :type a: ``np.ndarray``
    :param r:  the length of the combinations
    :type r: ``int``

    :returns: an array of combinations
    :rtype: ``np.ndarray``
    """
    # Special-case for 0-length combinations
    if r is 0:
        return np.asarray([])

    a = np.asarray(a)
    data_type = a.dtype if r is 0 else np.dtype([('', a.dtype)]*r)
    b = np.fromiter(combinations(a, r), data_type)
    return b.view(a.dtype).reshape(-1, r)


# see http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
def comb_indices(n, k):
    """
    N-D version of itertools.combinations.

    Return indices that yeild the :math:`r`-combinations of :math:`n` elements

        >>> n, k = 3, 2
        >>> data = np.arange(6).reshape(2, 3)
        >>> print(data)
        [[0 1 2]
         [3 4 5]]
        >>> print(data[:, comb_indices(n, k)])
        [[[0 1]
          [0 2]
          [1 2]]
         [[3 4]
          [3 5]
          [4 5]]]

    :param a: array from which to get combinations
    :type a: ``np.ndarray``
    :param k: length of combinations
    :type k: ``int``

    :returns: indices of the :math:`r`-combinations of :math:`n` elements
    :rtype: ``np.ndarray``
    """
    # Count the number of combinations for preallocation
    count = comb(n, k, exact=True)
    # Get numpy iterable from ``itertools.combinations``
    indices = np.fromiter(
        chain.from_iterable(combinations(range(n), k)),
        int,
        count=count*k)
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


# TODO: implement this with numpy?
def powerset(iterable):
    """
    Return the power set of an iterable (see `itertools recipes
    <http://docs.python.org/2/library/itertools.html#recipes>`_).

        >>> ps = powerset(np.arange[2])
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
    (so there are is one dimension per node, the size of which is the number of
    states for that node).

    :param nodes: a set of indices of binary nodes
    :type nodes: ``np.ndarray``

    :returns: the uniform distribution over the set of nodes
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
    # Preserve number of dimensions so node indices still index into
    # the proper axis of the returned distribution
    prenormalized = np.expand_dims(np.sum(tpm, node.index), node.index)
    # Normalize the distribution by number of states
    return np.divide(prenormalized, tpm.shape[node.index])

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
        raise ValidationException("Connectivity matrix must be square.")
