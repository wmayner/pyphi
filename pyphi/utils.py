#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Functions used by more than one PyPhi module or class, or that might be of
external use.
"""

import hashlib
import os
from itertools import chain, combinations, product
from time import time

import decorator
import numpy as np
from scipy.misc import comb

from . import config, constants


def state_of(nodes, network_state):
    """Return the state-tuple of the given nodes."""
    return tuple(network_state[n] for n in nodes) if nodes else ()


def all_states(n, big_endian=False):
    """Return all binary states for a system.

    Args:
        n (int): The number of elements in the system.
        big_endian (bool): Whether to return the states in big-endian order
            instead of little-endian order.

    Yields:
        tuple[int]: The next state of an ``n``-element system, in little-endian
        order unless ``big_endian`` is ``True``.
    """
    if n == 0:
        return

    for state in product((0, 1), repeat=n):
        if big_endian:
            yield state
        else:
            yield state[::-1]  # Convert to little-endian ordering


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


class np_hashable:
    """A hashable wrapper around a NumPy array."""
    # pylint: disable=protected-access

    def __init__(self, array):
        self._array = np_immutable(array.copy())

    def __hash__(self):
        return np_hash(self._array)

    def __eq__(self, other):
        return np.array_equal(self._array, other._array)

    def __repr__(self):
        return repr(self._array)


def eq(x, y):
    """Compare two values up to |PRECISION|."""
    return abs(x - y) <= constants.EPSILON


# see http://stackoverflow.com/questions/16003217
def combs(a, r):
    """NumPy implementation of ``itertools.combinations``.

    Return successive ``r``-length combinations of elements in the array ``a``.

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
    """``n``-dimensional version of itertools.combinations.

    Args:
        a (np.ndarray): The array from which to get combinations.
        k (int): The desired length of the combinations.

    Returns:
        np.ndarray: Indices that give the ``k``-combinations of ``n`` elements.

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


# From https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable, nonempty=False, reverse=False):
    """Generate the power set of an iterable.

    Args:
        iterable (Iterable): The iterable from which to generate the power set.

    Keyword Args:
        nonempty (boolean): If True, don't include the empty set.
        reverse (boolean): If True, reverse the order of the powerset.

    Returns:
        Iterable: An iterator over the power set.

    Example:
        >>> ps = powerset(np.arange(2))
        >>> list(ps)
        [(), (0,), (1,), (0, 1)]
        >>> ps = powerset(np.arange(2), nonempty=True)
        >>> list(ps)
        [(0,), (1,), (0, 1)]
        >>> ps = powerset(np.arange(2), nonempty=True, reverse=True)
        >>> list(ps)
        [(1, 0), (1,), (0,)]
    """
    iterable = list(iterable)

    if nonempty:  # Don't include 0-length subsets
        start = 1
    else:
        start = 0

    seq_sizes = range(start, len(iterable) + 1)

    if reverse:
        seq_sizes = reversed(seq_sizes)
        iterable.reverse()

    return chain.from_iterable(combinations(iterable, r) for r in seq_sizes)


def load_data(directory, num):
    """Load numpy data from the data directory.

    The files should stored in ``../data/<dir>`` and named
    ``0.npy, 1.npy, ... <num - 1>.npy``.

    Returns:
        list: A list of loaded data, such that ``list[i]`` contains the the
        contents of ``i.npy``.
    """
    root = os.path.abspath(os.path.dirname(__file__))

    def get_path(i):  # pylint: disable=missing-docstring
        return os.path.join(root, 'data', directory, str(i) + '.npy')

    return [np.load(get_path(i)) for i in range(num)]


# Using ``decorator`` preserves the function signature of the wrapped function,
# allowing joblib to properly introspect the function arguments.
@decorator.decorator
def time_annotated(func, *args, **kwargs):
    """Annotate the decorated function or method with the total execution
    time.

    The result is annotated with a `time` attribute.
    """
    start = time()
    result = func(*args, **kwargs)
    end = time()
    result.time = round(end - start, config.PRECISION)
    return result
