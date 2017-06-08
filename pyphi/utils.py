#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Functions used by more than one PyPhi module or class, or that might be of
external use.
"""

import os
import hashlib
from itertools import chain, combinations, product

import numpy as np
from scipy.misc import comb

from . import constants


def state_of(nodes, network_state):
    """Return the state-tuple of the given nodes."""
    return tuple(network_state[n] for n in nodes) if nodes else ()


def all_states(n, holi=False):
    """Return all binary states for a system.

    Args:
        n (int): The number of elements in the system.
        holi (bool): Whether to return the states in HOLI order instead of LOLI
            order.

    Yields:
        tuple[int]: The next state of an ``n``-element system, in LOLI order
            unless ``holi`` is ``True``.
    """
    if n == 0:
        return

    for state in product((0, 1), repeat=n):
        if holi:
            yield state
        else:
            yield state[::-1]  # Convert to LOLI-ordering


# TPM and Connectivity Matrix utils
# ============================================================================

def state_by_state(tpm):
    """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
    ``False``."""
    return tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]


def condition_tpm(tpm, fixed_nodes, state):
    """Return a TPM conditioned on the given fixed node indices, whose states
    are fixed according to the given state-tuple.

    The dimensions of the new TPM that correspond to the fixed nodes are
    collapsed onto their state, making those dimensions singletons suitable for
    broadcasting. The number of dimensions of the conditioned TPM will be the
    same as the unconditioned TPM.
    """
    conditioning_indices = [[slice(None)]] * len(state)
    for i in fixed_nodes:
        # Preserve singleton dimensions with `np.newaxis`
        conditioning_indices[i] = [state[i], np.newaxis]
    # Flatten the indices.
    conditioning_indices = list(chain.from_iterable(conditioning_indices))
    # Obtain the actual conditioned TPM by indexing with the conditioning
    # indices.
    return tpm[conditioning_indices]


def expand_tpm(tpm):
    """Broadcast a state-by-node TPM so that singleton dimensions are expanded
    over the full network."""
    uc = np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]])
    return tpm * uc


def apply_boundary_conditions_to_cm(external_indices, cm):
    """Return a connectivity matrix with all connections to or from external
    nodes removed.
    """
    cm = cm.copy()
    for i in external_indices:
        # Zero-out row
        cm[i] = 0
        # Zero-out column
        cm[:, i] = 0
    return cm


def get_inputs_from_cm(index, cm):
    """Return a tuple of node indices that have connections to the node with
    the given index.
    """
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])


def get_outputs_from_cm(index, cm):
    """Return a tuple of node indices that the node with the given index has
    connections to.
    """
    return tuple(i for i in range(cm.shape[0]) if cm[index][i])


def causally_significant_nodes(cm):
    """Return a tuple of all nodes indices in the connectivity matrix which
    are causally significant (have inputs and outputs)."""
    inputs = cm.sum(0)
    outputs = cm.sum(1)
    nodes_with_inputs_and_outputs = np.logical_and(inputs > 0, outputs > 0)
    return tuple(np.where(nodes_with_inputs_and_outputs)[0])


def np_immutable(a):
    """Make a NumPy array immutable."""
    a.flags.writeable = False
    return a


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
        np.ndarray: An array of combinations.
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
    """|N-D| version of itertools.combinations.

    Args:
        a (np.ndarray): The array from which to get combinations.
        k (int): The desired length of the combinations.

    Returns:
        np.ndarray: Indices that give the |k|-combinations of |n| elements.

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
        generator: An chained generator over the power set.

    Example:
        >>> ps = powerset(np.arange(2))
        >>> print(list(ps))
        [(), (0,), (1,), (0, 1)]
    """
    return chain.from_iterable(combinations(iterable, r)
                               for r in range(len(iterable) + 1))


def marginalize_out(indices, tpm):
    """
    Marginalize out a node from a TPM.

    Args:
        indices (list[int]): The indices of nodes to be marginalized out.
        tpm (np.ndarray): The TPM to marginalize the node out of.

    Returns:
        np.ndarray: A TPM with the same number of dimensions, with the nodes
        marginalized out.
    """
    return tpm.sum(tuple(indices), keepdims=True) / (
        np.array(tpm.shape)[list(indices)].prod())


def load_data(directory, num):
    """Load numpy data from the data directory.

    The files should stored in ``../data/{dir}`` and named
    ``0.npy, 1.npy, ... {num - 1}.npy``.

    Returns:
        list: A list of loaded data, such that ``list[i]`` contains the the
        contents of ``i.npy``.
    """

    root = os.path.abspath(os.path.dirname(__file__))

    def get_path(i):  # pylint: disable=missing-docstring
        return os.path.join(root, 'data', directory, str(i) + '.npy')

    return [np.load(get_path(i)) for i in range(num)]
