# utils.py
"""Package-wide utilities."""

import functools
import hashlib
import math
import operator
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import chain
from itertools import combinations
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import comb

from .conf import config


# TODO(states) refactor
def substate(
    nodes: tuple[int, ...], state: tuple[int, ...], node_subset: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the state restricted to ``node_subset`` using ``nodes`` indexing."""
    return tuple(state[nodes.index(n)] for n in node_subset)


def state_of(
    nodes: tuple[int, ...], substrate_state: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the state-tuple of the given nodes."""
    return tuple(substrate_state[n] for n in nodes) if nodes else ()


def state_of_system_nodes(
    node_indices: tuple[int, ...],
    nodes: tuple[int, ...],
    system_state: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the state of the nodes, given a system state-tuple.

    Deals with using the substrate-relative node indices nodes with a state-tuple
    for only the system nodes.
    """
    # Get indices relative to system indices
    return state_of(tuple(node_indices.index(n) for n in nodes), system_state)


def all_states(
    spec: int | Sequence[int],
    big_endian: bool = False,
) -> Generator[tuple[int, ...]]:
    """Return all states for a system.

    Args:
        spec: Either an integer ``n`` (binary, ``n`` nodes) or a sequence of
            per-node alphabet sizes.
        big_endian: Return states in big-endian order if ``True``, otherwise
            little-endian (index 0 varies fastest).

    Yields:
        tuple[int, ...]: Each possible state.

    Examples:
        Binary, 2 nodes (little-endian):

        >>> from pyphi.utils import all_states
        >>> list(all_states(2))
        [(0, 0), (1, 0), (0, 1), (1, 1)]

        Ternary first node, binary second (little-endian):

        >>> list(all_states((3, 2)))
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    """
    if isinstance(spec, int):
        alphabet_sizes: tuple[int, ...] = (2,) * spec
    else:
        alphabet_sizes = tuple(spec)

    if not alphabet_sizes:
        return

    ranges = [range(k) for k in alphabet_sizes]
    if big_endian:
        for state in product(*ranges):
            yield state
    else:
        for state in product(*reversed(ranges)):
            yield state[::-1]


def np_immutable(a: np.ndarray) -> np.ndarray:
    """Make a NumPy array immutable."""
    a.flags.writeable = False
    return a


def np_hash(a: np.ndarray | None) -> int:
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)  # pyright: ignore[reportOptionalMemberAccess]


class np_hashable:
    """A hashable wrapper around a NumPy array."""

    # pylint: disable=protected-access

    def __init__(self, array: np.ndarray) -> None:
        self._array = np_immutable(array.copy())

    def __hash__(self) -> int:
        return np_hash(self._array)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, np_hashable):
            return NotImplemented
        return np.array_equal(self._array, other._array)

    def __repr__(self) -> str:
        return repr(self._array)


def eq(x: float, y: float) -> bool:
    """Compare two values up to |PRECISION|."""
    # TODO(4.0) just use float value in config
    precision = int(config.numerics.precision)
    epsilon = 10 ** (-precision)
    return math.isclose(x, y, rel_tol=epsilon, abs_tol=epsilon)


def is_positive(x: float) -> bool:
    """Return whether ``x`` is positive up to |PRECISION|."""
    # Need `bool` to cast from numpy to native Boolean
    return not eq(x, 0) and bool(x > 0)


def is_nonpositive(x: float) -> bool:
    """Return True if x is a nonpositive value."""
    # Need `bool` to cast from numpy to native Boolean
    return bool(x <= 0)


def is_falsy(x: object) -> bool:
    """Return True if x is a falsy value."""
    return not x


def positive_part(x: float) -> float:
    """Return ``max(0, x)``.

    The ``|·|+`` operator from Eqs. 19-20 of the IIT 4.0 paper. Clamps a
    raw integration value to zero from below; PyPhi retains the raw
    signed value as metadata for "preventative cause" visibility while
    exposing the clamped value as ``φ``.
    """
    return max(0.0, float(x))


# see http://stackoverflow.com/questions/16003217
def combs(a: np.ndarray, r: int) -> np.ndarray:
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
def comb_indices(n: int, k: int) -> np.ndarray:
    """Return indices that generate the ``k``-combinations of ``n`` elements.

    Args:
        n (int): The total number of elements to choose from.
        k (int): The length of each combination.

    Returns:
        np.ndarray: A ``(comb(n, k), k)`` array of indices that can be used to
        select every length-``k`` combination from an array.

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
def powerset(
    iterable: Iterable[Any],
    nonempty: bool = False,
    reverse: bool = False,
    min_size: int = 0,
    max_size: int | None = None,
) -> chain[Any]:
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
        >>> ps = powerset(range(2))
        >>> list(ps)
        [(), (0,), (1,), (0, 1)]
        >>> ps = powerset(range(2), nonempty=True)
        >>> list(ps)
        [(0,), (1,), (0, 1)]
        >>> ps = powerset(range(2), nonempty=True, reverse=True)
        >>> list(ps)
        [(1, 0), (1,), (0,)]
        >>> ps = powerset(range(3), max_size=2)
        >>> list(ps)
        [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
        >>> ps = powerset(range(3), min_size=2)
        >>> list(ps)
        [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
        >>> ps = powerset(range(3), min_size=2, max_size=2)
        >>> list(ps)
        [(0, 1), (0, 2), (1, 2)]
    """
    iterable = list(iterable)

    if nonempty and min_size <= 0:  # Don't include 0-length subsets
        min_size = 1

    if max_size is None:
        max_size = len(iterable)

    _seq_sizes = range(min_size, max_size + 1)

    if reverse:
        seq_sizes: Iterable[int] = reversed(_seq_sizes)
        iterable.reverse()
    else:
        seq_sizes = _seq_sizes

    return chain.from_iterable(combinations(iterable, r) for r in seq_sizes)


def load_data(directory: str, num: int) -> list[np.ndarray]:
    """Load numpy data from the data directory.

    The files should stored in ``../data/<dir>`` and named
    ``0.npy, 1.npy, ... <num - 1>.npy``.

    Returns:
        list: A list of loaded data, such that ``list[i]`` contains the the
        contents of ``i.npy``.
    """
    root = Path(__file__).parent.resolve()

    def get_path(i: int) -> Path:  # pylint: disable=missing-docstring
        return root / "data" / directory / f"{i}.npy"

    return [np.load(get_path(i), allow_pickle=True) for i in range(num)]


def specified_substate(
    purview: tuple[int, ...], specified_state: np.ndarray, subset: tuple[int, ...]
) -> np.ndarray:
    """Return the specified state restricted to a subset of purview nodes."""
    purview_relative_subset = [purview.index(node) for node in subset]
    return specified_state[:, purview_relative_subset]


def extremum_with_short_circuit(
    seq: list,
    value_func: Callable = lambda item: item.phi,
    cmp: Callable = operator.lt,
    initial: float = float("inf"),
    shortcircuit_value: float = 0,
    shortcircuit_callback: Callable | None = None,
) -> object | None:
    """Return the extreme item in ``seq``, optionally short-circuiting early.

    Args:
        seq (Iterable): Items to evaluate.
        value_func (callable): Function extracting the value to compare from an
            item. Defaults to ``lambda item: item.phi``.
        cmp (callable): Comparison operator used to track the extremum; use
            ``operator.lt`` for minima or ``operator.gt`` for maxima.
        initial (float): Initial comparison value for the extremum tracker.
        shortcircuit_value (float): If ``value_func(item)`` equals this, return
            the item immediately.
        shortcircuit_callback (callable | None): Callback invoked when
            short-circuiting, if provided.

    Returns:
        object: The item with the extreme value according to ``cmp``.
    """
    extreme_item: object | None = None
    extreme_value: float = initial
    for item in seq:
        value = value_func(item)
        if value == shortcircuit_value:
            if shortcircuit_callback is not None:
                shortcircuit_callback()
            return item  # type: ignore[no-any-return]
        if cmp(value, extreme_value):
            extreme_value = value
            extreme_item = item
    return extreme_item


def expsublog(x: float, y: float) -> float:
    """Computes ``x / y`` as ``exp(log(x) - log(y))``.

    Useful for dividing by extremely large denominators.

    See also ``numpy.logaddexp``.
    """
    return math.exp(math.log(x) - math.log(y))


def expaddlog(x: float, y: float) -> float:
    """Computes ``x * y`` as ``exp(log(x) + log(y))``.

    Useful for dividing by extremely large denominators.

    See also ``numpy.logaddexp``.
    """
    return math.exp(math.log(x) + math.log(y))


def _try_len(iterable: object) -> int | None:
    """Return ``len(iterable)`` if available, otherwise ``None``."""
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


def try_len(*iterables: object) -> int | None:
    """Return the minimum length of iterables, or ``None`` if none have a length."""
    lengths = (_try_len(it) for it in iterables)
    return min((length for length in lengths if length is not None), default=None)


def assume_integer(x: float) -> int:
    """Attempt cast to integer, raising an error if it is not an integer."""
    if isinstance(x, float) and not x.is_integer():
        raise ValueError(f"expected integer, got {type(x)} {x}")
    return int(x)


def enforce_integer(i: int, name: str = "", min: float = float("-inf")) -> int:
    """Ensure ``i`` is an int not less than ``min``, raising on violation."""
    if not isinstance(i, int) or i < min:
        raise ValueError(f"{name} must be a positive integer")
    return i


def enforce_integer_or_none(i: int | None, **kwargs: str | float) -> int | None:
    """Validate ``i`` as an integer or pass through ``None``."""
    if i is None:
        return i
    return enforce_integer(i, **kwargs)  # type: ignore[arg-type]


def all_same(comparison: Callable, seq: Generator | list) -> bool:
    """Return True if all elements compare to the first element."""
    sentinel = object()
    first = next(iter(seq), sentinel)
    if first is sentinel:
        # Vacuously
        return True
    return all(comparison(first, other) for other in seq)


# Compare equality up to precision
all_are_equal = functools.partial(all_same, eq)
all_are_identical = functools.partial(all_same, operator.is_)


NO_DEFAULT = object()


# TODO test
def all_extrema(
    comparison: Callable, seq: Generator | list, default: object = NO_DEFAULT
) -> list:
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
    extrema: list = []
    sentinel = object()
    current_extremum = next(iter(seq), sentinel)
    if current_extremum is sentinel:
        if default is NO_DEFAULT:
            raise ValueError("Cannot find extrema of empty sequence without default")
        return [default]
    extrema.append(current_extremum)
    for element in seq:
        if comparison(element, current_extremum):
            extrema = [element]
            current_extremum = element
        elif element == current_extremum:
            extrema.append(element)
    return extrema


all_minima = functools.partial(all_extrema, operator.lt)
all_maxima = functools.partial(all_extrema, operator.gt)


def iter_with_default(seq: Iterable[Any], default: object) -> Generator[Any]:
    """Iterate over ``seq``, yielding ``default`` if ``seq`` is empty."""
    yielded = False
    for item in seq:
        yield item
        yielded = True
    if not yielded:
        if default is NO_DEFAULT:
            raise ValueError("Cannot iterate over empty sequence without default")
        yield default
