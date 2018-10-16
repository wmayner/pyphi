#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distribution.py

"""
Functions for manipulating probability distributions.
"""

import numpy as np

from .cache import cache


def normalize(a):
    """Normalize a distribution.

    Args:
        a (np.ndarray): The array to normalize.

    Returns:
        np.ndarray: ``a`` normalized so that the sum of its entries is 1.
    """
    sum_a = a.sum()
    if sum_a == 0:
        return a
    return a / sum_a


# TODO? remove this? doesn't seem to be used anywhere
def uniform_distribution(number_of_nodes):
    """
    Return the uniform distribution for a set of binary nodes, indexed by state
    (so there is one dimension per node, the size of which is the number of
    possible states for that node).

    Args:
        nodes (np.ndarray): A set of indices of binary nodes.

    Returns:
        np.ndarray: The uniform distribution over the set of nodes.
    """
    # The size of the state space for binary nodes is 2^(number of nodes).
    number_of_states = 2 ** number_of_nodes
    # Generate the maximum entropy distribution
    # TODO extend to nonbinary nodes
    return (np.ones(number_of_states) /
            number_of_states).reshape([2] * number_of_nodes)


def marginal_zero(repertoire, node_index):
    """Return the marginal probability that the node is OFF."""
    index = [slice(None)] * repertoire.ndim
    index[node_index] = 0

    return repertoire[tuple(index)].sum()


def marginal(repertoire, node_index):
    """Get the marginal distribution for a node."""
    index = tuple(i for i in range(repertoire.ndim) if i != node_index)

    return repertoire.sum(index, keepdims=True)


def independent(repertoire):
    """Check whether the repertoire is independent."""
    marginals = [marginal(repertoire, i) for i in range(repertoire.ndim)]

    # TODO: is there a way to do without an explicit iteration?
    joint = marginals[0]
    for m in marginals[1:]:
        joint = joint * m

    # TODO: should we round here?
    # repertoire = repertoire.round(config.PRECISION)
    # joint = joint.round(config.PRECISION)

    return np.array_equal(repertoire, joint)


def purview(repertoire):
    """The purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        tuple[int]: The purview that the repertoire was computed over.
    """
    if repertoire is None:
        return None

    return tuple(i for i, dim in enumerate(repertoire.shape) if dim == 2)


def purview_size(repertoire):
    """Return the size of the purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        int: The size of purview that the repertoire was computed over.
    """
    return len(purview(repertoire))


def repertoire_shape(purview, N):  # pylint: disable=redefined-outer-name
    """Return the shape a repertoire.

    Args:
        purview (tuple[int]): The purview over which the repertoire is
            computed.
        N (int): The number of elements in the system.

    Returns:
        list[int]: The shape of the repertoire. Purview nodes have two
        dimensions and non-purview nodes are collapsed to a unitary dimension.

    Example:
        >>> purview = (0, 2)
        >>> N = 3
        >>> repertoire_shape(purview, N)
        [2, 1, 2]
    """
    # TODO: extend to non-binary nodes
    return [2 if i in purview else 1 for i in range(N)]


def flatten(repertoire, big_endian=False):
    """Flatten a repertoire, removing empty dimensions.

    By default, the flattened repertoire is returned in little-endian order.

    Args:
        repertoire (np.ndarray or None): A repertoire.

    Keyword Args:
        big_endian (boolean): If ``True``, flatten the repertoire in big-endian
            order.

    Returns:
        np.ndarray: The flattened repertoire.
    """
    if repertoire is None:
        return None

    order = 'C' if big_endian else 'F'
    # For efficiency, use `ravel` (which returns a view of the array) instead
    # of `np.flatten` (which copies the whole array).
    return repertoire.squeeze().ravel(order=order)


@cache(cache={}, maxmem=None)
def max_entropy_distribution(node_indices, number_of_nodes):
    """Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside ``node_indices`` are fixed and treated as if they have only 1
    state.

    Args:
        node_indices (tuple[int]): The set of node indices over which to take
            the distribution.
        number_of_nodes (int): The total number of nodes in the network.

    Returns:
        np.ndarray: The maximum entropy distribution over the set of nodes.
    """
    distribution = np.ones(repertoire_shape(node_indices, number_of_nodes))

    return distribution / distribution.size
