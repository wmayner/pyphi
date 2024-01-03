# utils.py

"""
Functions used by more than one PyPhi module or class, or that might be of
external use.
"""

import hashlib
import math
import operator
import os
from itertools import chain, combinations, product
from typing import Tuple

import numpy as np
from scipy.special import comb
from toolz import curry

from .conf import config


# TODO(states) refactor
def substate(
    nodes: Tuple[int], state: Tuple[int], node_subset: Tuple[int]
) -> Tuple[int]:
    return tuple(state[nodes.index(n)] for n in node_subset)


def state_of(nodes, network_state):
    """Return the state-tuple of the given nodes."""
    return tuple(network_state[n] for n in nodes) if nodes else ()


def state_of_subsystem_nodes(node_indices, nodes, subsystem_state):
    """Return the state of the nodes, given a subsystem state-tuple.

    Deals with using the network-relative node indices nodes with a state-tuple
    for only the subsystem nodes.
    """
    # Get indices relative to subsystem indices
    return state_of([node_indices.index(n) for n in nodes], subsystem_state)


# TODO: nonbinary states
def all_states(n, big_endian=False):
    """Return all binary states for a system.

    Args:
        n (int): The number of elements in the system.
        big_endian (bool): Whether to return the states in big-endian order
            instead of little-endian order.

    Yields:
        Tuple[int]: The next state of an ``n``-element system, in little-endian
        order unless ``big_endian`` is ``True``.
    """
    if n == 0:
        return

    for state in product((0, 1), repeat=n):
        if big_endian:
            yield state
        else:
            yield state[::-1]  # Convert to little-endian ordering


def equivalent_states(state, mask, subsystem):
    """Generate equivalence class of states given irrelevant dimensions.

    Arguments:
        state (Iterable[int]): Some state in the equivalence class.
        mask (Iterable[int]): State mask with 1's representing irrelevant dimensions.
        subsystem (|Subsystem|): The subsystem of interest.

    Yields:
        Iterable[tuple[int]]: A generator for the equivalence class of states.

    Examples:
        >>> import numpy as np
        >>> from pyphi import Network, Subsystem
        >>> network = Network(np.ones((16, 4)))
        >>> subsystem = Subsystem(network, (1, 1, 1, 1))
        >>> state = (1, 1, 1, 1)
        >>> mask = (2, 1, 1, 2)
        >>> list(equivalent_states(state, mask, subsystem))
        [(1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 1)]
    """
    indices_needing_expansion = {
        i: subsystem.tpm.shape[i] for i in subsystem.node_indices
        if mask[i] == 1
    }
    locally_expanded_states = product(
        *[range(states) for i, states in indices_needing_expansion.items()]
    )
    expanded_indices = list(indices_needing_expansion.keys())
    state = np.array(state)
    for s in locally_expanded_states:
        state[expanded_indices] = s
        yield tuple(state)


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
    # TODO(4.0) just use float value in config
    epsilon = 10 ** (-config.PRECISION)
    return math.isclose(x, y, rel_tol=epsilon, abs_tol=epsilon)


def is_positive(x):
    """Return whether ``x`` is positive up to |PRECISION|."""
    # Need `bool` to cast from numpy to native Boolean
    return not eq(x, 0) and bool(x > 0)


def is_nonpositive(x):
    """Return True if x is a nonpositive value."""
    # Need `bool` to cast from numpy to native Boolean
    return bool(x <= 0)


def is_falsy(x):
    """Return True if x is a falsy value."""
    return not x


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
    data_type = a.dtype if r == 0 else np.dtype([("", a.dtype)] * r)
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
        chain.from_iterable(combinations(range(n), k)), int, count=(count * k)
    )
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


# Based on https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable, nonempty=False, reverse=False, min_size=0, max_size=None):
    """Generate the power set of an iterable.

    Args:
        iterable (Iterable): The iterable of which to generate the power set.

    Keyword Args:
        nonempty (boolean): If True, don't include the empty set.
        reverse (boolean): If True, reverse the order of the powerset.
        min_size (int | None): Only generate subsets of this size or larger.
            Defaults to None, meaning no restriction. Overrides ``nonempty``.
        max_size (int | None): Only generate subsets of this size or smaller.
            Defaults to None, meaning no restriction.

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
        >>> ps = powerset(np.arange(3), max_size=2)
        >>> list(ps)
        [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
        >>> ps = powerset(np.arange(3), min_size=2)
        >>> list(ps)
        [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
        >>> ps = powerset(np.arange(3), min_size=2, max_size=2)
        >>> list(ps)
        [(0, 1), (0, 2), (1, 2)]
    """
    iterable = list(iterable)

    if nonempty and min_size <= 0:  # Don't include 0-length subsets
        min_size = 1

    if max_size is None:
        max_size = len(iterable)

    seq_sizes = range(min_size, max_size + 1)

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
        return os.path.join(root, "data", directory, str(i) + ".npy")

    return [np.load(get_path(i), allow_pickle=True) for i in range(num)]


def specified_substate(purview, specified_state, subset):
    purview_relative_subset = [purview.index(node) for node in subset]
    return specified_state[:, purview_relative_subset]


def extremum_with_short_circuit(
    seq,
    value_func=lambda item: item.phi,
    cmp=operator.lt,
    initial=float("inf"),
    shortcircuit_value=0,
    shortcircuit_callback=None,
):
    """Return the extreme value, optionally shortcircuiting."""
    extreme_item = None
    extreme_value = initial
    for item in seq:
        value = value_func(item)
        if value == shortcircuit_value:
            try:
                shortcircuit_callback()
            except TypeError:
                pass
            return item
        if cmp(value, extreme_value):
            extreme_value = value
            extreme_item = item
    return extreme_item


def expsublog(x, y):
    """Computes ``x / y`` as ``exp(log(x) - log(y))``.

    Useful for dividing by extremely large denominators.

    See also ``numpy.logaddexp``.
    """
    return math.exp(math.log(x) - math.log(y))


def expaddlog(x, y):
    """Computes ``x * y`` as ``exp(log(x) + log(y))``.

    Useful for dividing by extremely large denominators.

    See also ``numpy.logaddexp``.
    """
    return math.exp(math.log(x) + math.log(y))


def _try_len(iterable):
    try:
        return len(iterable)
    except TypeError:
        return None


def try_len(*iterables):
    """Return the minimum length of iterables, or ``None`` if none have a length."""
    lengths = (_try_len(it) for it in iterables)
    return min((l for l in lengths if l is not None), default=None)


def assume_integer(x):
    """Attempt cast to integer, raising an error if it is not an integer."""
    if isinstance(x, float) and not x.is_integer():
        raise ValueError(f"expected integer, got {type(x)} {x}")
    return int(x)


def enforce_integer(i, name="", min=float("-inf")):
    if not isinstance(i, int) or i < min:
        raise ValueError(f"{name} must be a positive integer")
    return i


def enforce_integer_or_none(i, **kwargs):
    if i is None:
        return i
    return enforce_integer(i, **kwargs)


@curry
def all_same(comparison, seq):
    sentinel = object()
    first = next(seq, sentinel)
    if first is sentinel:
        # Vacuously
        return True
    return all(comparison(first, other) for other in seq)


# Compare equality up to precision
all_are_equal = all_same(eq)
all_are_identical = all_same(operator.is_)


NO_DEFAULT = object()

# TODO test
@curry
def all_extrema(comparison, seq, default=NO_DEFAULT):
    """Return the extrema of ``seq``.

    Use ``<`` as the comparison to obtain the minima; use ``>`` as the
    comparison to obtain the maxima.

    Uses only one pass through ``seq``.

    Args:
        comparison (callable): A comparison operator.
        seq (iterator): An iterator over a sequence.

    Returns:
        list: The maxima/minima in ``seq``.
    """
    extrema = []
    sentinel = object()
    current_extremum = next(seq, sentinel)
    if current_extremum is sentinel:
        if default is NO_DEFAULT:
            raise ValueError("Cannot find extrema of empty sequence without default")
        else:
            return [default]
    extrema.append(current_extremum)
    for element in seq:
        if comparison(element, current_extremum):
            extrema = [element]
            current_extremum = element
        elif element == current_extremum:
            extrema.append(element)
    return extrema


all_minima = all_extrema(operator.lt)
all_maxima = all_extrema(operator.gt)


def iter_with_default(seq, default):
    """Iterate over ``seq``, yielding ``default`` if ``seq`` is empty."""
    yielded = False
    for item in seq:
        yield item
        yielded = True
    if not yielded:
        if default is NO_DEFAULT:
            raise ValueError("Cannot iterate over empty sequence without default")
        yield default
