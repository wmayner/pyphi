# distribution.py
"""Functions for manipulating probability distributions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .cache import cache
from .types import NodeIndices
from .types import Purview
from .types import Repertoire


def normalize(a: Repertoire) -> Repertoire:
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
def uniform_distribution(number_of_nodes: int) -> Repertoire:
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
    number_of_states = 2**number_of_nodes
    # Generate the maximum entropy distribution
    # TODO extend to nonbinary nodes
    return (np.ones(number_of_states) / number_of_states).reshape([2] * number_of_nodes)


def marginal_zero(repertoire: Repertoire, node_index: int) -> np.floating:
    """Return the marginal probability that the node is OFF."""
    index: list[slice | int] = [slice(None)] * repertoire.ndim
    index[node_index] = 0

    return repertoire[tuple(index)].sum()


def marginal(repertoire: Repertoire, node_index: int) -> Repertoire:
    """Get the marginal distribution for a node."""
    index = tuple(i for i in range(repertoire.ndim) if i != node_index)

    return repertoire.sum(index, keepdims=True)


def independent(repertoire: Repertoire) -> bool:
    """Check whether the repertoire is independent."""
    marginals = [marginal(repertoire, i) for i in range(repertoire.ndim)]

    # TODO: is there a way to do without an explicit iteration?
    joint = marginals[0]
    for m in marginals[1:]:
        joint = joint * m

    # TODO: should we round here?
    # repertoire = repertoire.round(config.numerics.precision)
    # joint = joint.round(config.numerics.precision)

    return bool(np.array_equal(repertoire, joint))


def purview(repertoire: Repertoire | None) -> Purview | None:
    """The purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        tuple[int]: The purview that the repertoire was computed over.
    """
    if repertoire is None:
        return None

    return tuple(i for i, dim in enumerate(repertoire.shape) if dim == 2)


def purview_size(repertoire: Repertoire | None) -> int:
    """Return the size of the purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        int: The size of purview that the repertoire was computed over.
    """
    p = purview(repertoire)
    if p is None:
        return 0
    return len(p)


def repertoire_shape(
    all_node_indices: NodeIndices | Iterable[int], purview: Purview | Iterable[int]
) -> list[int]:
    """Return the shape a repertoire.

    Args:
        all_node_indices (tuple[int]): The node indices of the network.
        purview (tuple[int]): The indices of nodes in the repertoire.

    Returns:
        list[int]: The shape of the repertoire. Purview nodes have two
        dimensions and non-purview nodes are collapsed to a unitary dimension.

    Example:
        >>> purview = (0, 2)
        >>> repertoire_shape(range(3), purview)
        [2, 1, 2]
    """
    # TODO: extend to non-binary nodes
    return [2 if i in purview else 1 for i in all_node_indices]


def flatten(
    repertoire: Repertoire | None, big_endian: bool = False
) -> Repertoire | None:
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

    order = "C" if big_endian else "F"
    # For efficiency, use `ravel` (which returns a view of the array) instead
    # of `np.flatten` (which copies the whole array).
    return repertoire.squeeze().ravel(order=order)


def unflatten(
    repertoire: Repertoire, purview: Purview, N: int, big_endian: bool = False
) -> Repertoire:
    """Unflatten a repertoire.

    By default, the input is assumed to be in little-endian order.

    Args:
        repertoire (np.ndarray or None): A probability distribution.
        purview (Iterable[int]): The indices of the nodes whose states the
            probability is distributed over.
        N (int): The size of the network.

    Keyword Args:
        big_endian (boolean): If ``True``, assume the flat repertoire is in
            big-endian order.

    Returns:
        np.ndarray: The unflattened repertoire.
    """
    order = "C" if big_endian else "F"
    return repertoire.reshape(repertoire_shape(range(N), purview), order=order)


@cache(cache={}, maxmem=None)
def max_entropy_distribution(
    all_node_indices: NodeIndices, purview: Purview
) -> Repertoire:
    """Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside ``node_indices`` are fixed and treated as if they have only 1
    state.

    Args:
        all_node_indices (tuple[int]): The node indices of the network.
        purview (tuple[int]): The indices of nodes in the distribution.

    Returns:
        np.ndarray: The maximum entropy distribution over the set of nodes.
    """
    distribution = np.ones(repertoire_shape(all_node_indices, purview))
    return distribution / distribution.size
