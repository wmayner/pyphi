# distribution.py
"""Functions for manipulating probability distributions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .cache import cache
from .types import NodeIndices
from .types import Purview
from .types import Repertoire
from .utils import np_immutable


def normalize(a: Repertoire) -> Repertoire:
    """Normalize a distribution.

    Parameters
    ----------
    a : np.ndarray
        The array to normalize.

    Returns
    -------
    np.ndarray
        ``a`` normalized so that the sum of its entries is 1. If the entries
        sum to 0, ``a`` is returned unchanged.
    """
    sum_a = a.sum()
    if sum_a == 0:
        return a
    return a / sum_a


# TODO? remove this? doesn't seem to be used anywhere
def uniform_distribution(number_of_nodes: int) -> Repertoire:
    """Return the uniform distribution over a set of binary nodes.

    The distribution is indexed by state, with one dimension per node of size 2
    (the number of states of a binary node).

    Parameters
    ----------
    number_of_nodes : int
        The number of binary nodes.

    Returns
    -------
    np.ndarray
        The uniform distribution over the ``2 ** number_of_nodes`` states,
        shaped with one length-2 dimension per node.
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
    """Return whether the repertoire factorizes into its single-node marginals.

    A repertoire is independent when it equals the outer product of its
    per-node marginal distributions.
    """
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
    """Return the purview over which a repertoire is distributed.

    Purview nodes are identified as those with a non-unitary axis: a purview
    node carries its full alphabet along its dimension (size ≥ 2), while a
    node outside the purview is collapsed to a unitary (size-1) dimension. This
    holds for k-ary as well as binary nodes.

    Parameters
    ----------
    repertoire : np.ndarray or None
        A repertoire, or ``None``.

    Returns
    -------
    tuple[int] or None
        The indices of the purview nodes, or ``None`` if ``repertoire`` is
        ``None``.
    """
    if repertoire is None:
        return None

    return tuple(i for i, dim in enumerate(repertoire.shape) if dim > 1)


def purview_size(repertoire: Repertoire | None) -> int:
    """Return the size of a repertoire's purview.

    Parameters
    ----------
    repertoire : np.ndarray or None
        A repertoire, or ``None``.

    Returns
    -------
    int
        The number of purview nodes, or 0 if ``repertoire`` is ``None``.
    """
    p = purview(repertoire)
    if p is None:
        return 0
    return len(p)


def repertoire_shape(
    all_node_indices: NodeIndices | Iterable[int],
    purview: Purview | Iterable[int],
    alphabet_sizes: tuple[int, ...] | None = None,
) -> list[int]:
    """Return the shape of a repertoire.

    Parameters
    ----------
    all_node_indices : tuple[int]
        The node indices of the substrate.
    purview : tuple[int]
        The indices of nodes in the repertoire.
    alphabet_sizes : tuple[int, ...] or None, optional
        Per-node alphabet sizes indexed by node index. When ``None`` (the
        default), all purview nodes are treated as binary (alphabet size 2).

    Returns
    -------
    list[int]
        The shape of the repertoire. Purview nodes take their alphabet size (or
        2 when binary) and non-purview nodes are collapsed to a unitary
        dimension.

    Examples
    --------
    >>> purview = (0, 2)
    >>> repertoire_shape(range(3), purview)
    [2, 1, 2]
    """
    purview_set = set(purview)
    if alphabet_sizes is None:
        return [2 if i in purview_set else 1 for i in all_node_indices]
    return [alphabet_sizes[i] if i in purview_set else 1 for i in all_node_indices]


def flatten(
    repertoire: Repertoire | None, big_endian: bool = False
) -> Repertoire | None:
    """Flatten a repertoire, removing empty dimensions.

    By default, the flattened repertoire is returned in little-endian order.

    Parameters
    ----------
    repertoire : np.ndarray or None
        A repertoire.
    big_endian : bool, optional
        If ``True``, flatten the repertoire in big-endian order.

    Returns
    -------
    np.ndarray or None
        The flattened repertoire, or ``None`` if ``repertoire`` is ``None``.
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

    Parameters
    ----------
    repertoire : np.ndarray
        A probability distribution.
    purview : Iterable[int]
        The indices of the nodes whose states the probability is distributed
        over.
    N : int
        The size of the substrate.
    big_endian : bool, optional
        If ``True``, assume the flat repertoire is in big-endian order.

    Returns
    -------
    np.ndarray
        The unflattened repertoire, shaped with one dimension per substrate
        node.
    """
    order = "C" if big_endian else "F"
    return repertoire.reshape(repertoire_shape(range(N), purview), order=order)


@cache(cache={}, maxmem=None)
def max_entropy_distribution(
    all_node_indices: NodeIndices,
    purview: Purview,
    alphabet_sizes: tuple[int, ...] | None = None,
) -> Repertoire:
    """Return the maximum entropy distribution over a purview.

    This differs from the substrate's uniform distribution in that nodes outside
    ``purview`` are held fixed and treated as if they have only one state (a
    collapsed dimension).

    Parameters
    ----------
    all_node_indices : tuple[int]
        The node indices of the substrate.
    purview : tuple[int]
        The indices of nodes the distribution is over.
    alphabet_sizes : tuple[int, ...] or None, optional
        Per-node alphabet sizes indexed by node index. When ``None``, all nodes
        are treated as binary.

    Returns
    -------
    np.ndarray
        The maximum entropy distribution, uniform over the states of the purview
        nodes. The array is cached and read-only.
    """
    distribution = np.ones(
        repertoire_shape(all_node_indices, purview, alphabet_sizes=alphabet_sizes)
    )
    return np_immutable(distribution / distribution.size)
