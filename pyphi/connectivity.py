# connectivity.py
"""Functions for determining network connectivity properties."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import connected_components


def subadjacency(
    cm: ArrayLike, source: tuple[int, ...], target: tuple[int, ...] | None = None
) -> NDArray[np.int_]:
    """Return the sub-adjacency matrix for two groups of nodes.

    This gives the connections from the first group to the second group.

    Arguments:
        source (Iterable[int]): The source nodes.

    Keyword Arguments:
        target (Iterable[int] | None): The target nodes. If ``None``,
            defaults to be the same as the first.

    Example:
        >>> cm = np.identity(5)
        >>> subadjacency(cm, (2, 3))
        array([[1., 0.],
               [0., 1.]])
        >>> subadjacency(cm, (0, 1), (1, 2))
        array([[0., 0.],
               [1., 0.]])
    """
    cm_array = np.asarray(cm)
    if target is None:
        target = source
    return cm_array[np.ix_(source, target)]


def apply_boundary_conditions_to_cm(
    external_indices: tuple[int, ...], cm: ArrayLike
) -> NDArray[np.int_]:
    """Remove connections to or from external nodes."""
    cm_array = np.asarray(cm).copy()
    cm_array[external_indices, :] = 0  # Zero-out row
    cm_array[:, external_indices] = 0  # Zero-out columnt
    return cm_array


def get_inputs_from_cm(index: int, cm: ArrayLike) -> tuple[int, ...]:
    """Return indices of inputs to the node with the given index."""
    cm_array = np.asarray(cm)
    return tuple(i for i in range(cm_array.shape[0]) if cm_array[i][index])


def get_outputs_from_cm(index: int, cm: ArrayLike) -> tuple[int, ...]:
    """Return indices of the outputs of node with the given index."""
    cm_array = np.asarray(cm)
    return tuple(i for i in range(cm_array.shape[0]) if cm_array[index][i])


def causally_significant_nodes(cm: ArrayLike) -> tuple[int, ...]:
    """Return indices of nodes that have both inputs and outputs."""
    cm_array = np.asarray(cm)
    inputs = cm_array.sum(0)
    outputs = cm_array.sum(1)
    nodes_with_inputs_and_outputs = np.logical_and(inputs > 0, outputs > 0)
    return tuple(np.where(nodes_with_inputs_and_outputs)[0])


# TODO: better name?
def relevant_connections(
    n: int, _from: tuple[int, ...], to: tuple[int, ...]
) -> NDArray[np.float64]:
    """Construct a connectivity matrix.

    Args:
        n (int): The dimensions of the matrix
        _from (tuple[int]): Nodes with outgoing connections to ``to``
        to (tuple[int]): Nodes with incoming connections from ``_from``

    Returns:
        np.ndarray: An |n x n| connectivity matrix with the |i,jth| entry is
        ``1`` if |i| is in ``_from`` and |j| is in ``to``, and 0 otherwise.
    """
    cm = np.zeros((n, n))

    # Don't try and index with empty arrays. Older versions of NumPy
    # (at least up to 1.9.3) break with empty array indices.
    if not _from or not to:
        return cm

    cm[np.ix_(_from, to)] = 1
    return cm


def block_cm(cm: ArrayLike) -> bool:
    """Return whether ``cm`` can be arranged as a block connectivity matrix.

    If so, the corresponding mechanism/purview is trivially reducible.
    Technically, only square matrices are "block diagonal", but the notion of
    connectivity carries over.

    We test for block connectivity by trying to grow a block of nodes such
    that:

    - 'source' nodes only input to nodes in the block
    - 'sink' nodes only receive inputs from source nodes in the block

    For example, the following connectivity matrix represents connections from
    ``nodes1 = A, B, C`` to ``nodes2 = D, E, F, G`` (without loss of
    generality, note that ``nodes1`` and ``nodes2`` may share elements)::

         D  E  F  G
      A [1, 1, 0, 0]
      B [1, 1, 0, 0]
      C [0, 0, 1, 1]

    Since nodes |AB| only connect to nodes |DE|, and node |C| only connects to
    nodes |FG|, the subgraph is reducible, because the cut ::

      A,B    C
      ─── ✕ ───
      D,E   F,G

    does not change the structure of the graph.
    """
    cm_array = np.asarray(cm)
    if np.any(cm_array.sum(1) == 0):
        return True
    if np.all(cm_array.sum(1) == 1):
        return True

    outputs = list(range(cm_array.shape[1]))

    # CM helpers:
    def outputs_of(nodes: list[int] | NDArray[np.intp]) -> NDArray[np.intp]:
        """Return all nodes that `nodes` connect to (output to)."""
        return np.where(cm_array[nodes, :].sum(0))[0]

    def inputs_to(nodes: list[int] | NDArray[np.intp]) -> NDArray[np.intp]:
        """Return all nodes which connect to (input to) `nodes`."""
        return np.where(cm_array[:, nodes].sum(1))[0]

    # Start: source node with most outputs
    sources = [int(np.argmax(cm_array.sum(1)))]
    sinks = outputs_of(sources)
    sink_inputs = inputs_to(sinks)

    while True:
        if np.array_equal(sink_inputs, sources):
            # sources exclusively connect to sinks.
            # There are no other nodes which connect sink nodes,
            # hence set(sources) + set(sinks) form a component
            # which is not connected to the rest of the graph
            return True

        # Recompute sources, sinks, and sink_inputs
        sources = list(sink_inputs)  # Convert NDArray to list
        sinks = outputs_of(sources)
        sink_inputs = inputs_to(sinks)

        # Considering all output nodes?
        if np.array_equal(sinks, outputs):
            return False


# TODO: simplify the conditional validation here and in block_cm
# TODO: combine with fully_connected
def block_reducible(
    cm: ArrayLike, nodes1: tuple[int, ...], nodes2: tuple[int, ...]
) -> bool:
    """Return whether connections from ``nodes1`` to ``nodes2`` are reducible.

    Args:
        cm (np.ndarray): The network's connectivity matrix.
        nodes1 (tuple[int]): Source nodes
        nodes2 (tuple[int]): Sink nodes
    """
    # Trivial case
    if not nodes1 or not nodes2:
        return True

    cm_array = np.asarray(cm)
    cm_sub = cm_array[np.ix_(nodes1, nodes2)]

    # Validate the connectivity matrix.
    if not cm_sub.sum(0).all() or not cm_sub.sum(1).all():
        return True
    if len(nodes1) > 1 and len(nodes2) > 1:
        return block_cm(cm_sub)
    return False


def _connected(cm: ArrayLike, nodes: tuple[int, ...] | None, connection: str) -> bool:
    """Test connectivity for the connectivity matrix."""
    cm_to_check: ArrayLike = subadjacency(cm, nodes) if nodes is not None else cm

    num_components, _ = connected_components(cm_to_check, connection=connection)
    return bool(num_components < 2)


def is_strong(cm: ArrayLike, nodes: tuple[int, ...] | None = None) -> bool:
    """Return whether the connectivity matrix is strongly connected.

    Remember that a singleton graph is strongly connected.

    Args:
        cm (np.ndarray): A square connectivity matrix.

    Keyword Args:
        nodes (tuple[int]): A subset of nodes to consider.
    """
    return _connected(cm, nodes, "strong")


def is_weak(cm: ArrayLike, nodes: tuple[int, ...] | None = None) -> bool:
    """Return whether the connectivity matrix is weakly connected.

    Args:
        cm (np.ndarray): A square connectivity matrix.

    Keyword Args:
        nodes (tuple[int]): A subset of nodes to consider.
    """
    return _connected(cm, nodes, "weak")


def is_full(cm: ArrayLike, nodes1: tuple[int, ...], nodes2: tuple[int, ...]) -> bool:
    """Test connectivity of one set of nodes to another.

    Args:
        cm (``np.ndarrray``): The connectivity matrix
        nodes1 (tuple[int]): The nodes whose outputs to ``nodes2`` will be
            tested.
        nodes2 (tuple[int]): The nodes whose inputs from ``nodes1`` will
            be tested.

    Returns:
        bool: ``True`` if all elements in ``nodes1`` output to some element in
        ``nodes2`` and all elements in ``nodes2`` have an input from some
        element in ``nodes1``, or if either set of nodes is empty; ``False``
        otherwise.
    """
    if not nodes1 or not nodes2:
        return True

    cm_sub = subadjacency(cm, nodes1, nodes2)

    # Do all nodes have at least one connection?
    return bool(cm_sub.sum(0).all() and cm_sub.sum(1).all())
