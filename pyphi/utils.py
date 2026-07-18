# utils.py
"""Package-wide utilities."""

import hashlib
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import chain
from itertools import combinations
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


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

    Parameters
    ----------
    spec : int or Sequence[int]
        Either an integer ``n`` (binary system, ``n`` nodes) or a sequence of
        per-node alphabet sizes.
    big_endian : bool
        Return states in big-endian order if ``True``, otherwise little-endian
        (index 0 varies fastest).

    Yields
    ------
    tuple[int, ...]
        Each possible state.

    Examples
    --------
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


def is_falsy(x: object) -> bool:
    """Return True if x is a falsy value."""
    return not x


def positive_part(x: float) -> float:
    """Return ``max(0, x)``, the positive-part operator ``|·|⁺``.

    Rectifies a signed integration value by setting negative values to zero.
    In IIT 4.0 this operator appears in the definitions of the integrated
    effect and cause information φ (Albantakis et al., 2023, Eqs. 19 and 20):
    a partition that lowers the probability of the effect or cause state
    yields a negative value, which is clamped to zero so that only genuine
    increases in probability contribute.
    """
    return max(0.0, float(x))


# see http://stackoverflow.com/questions/16003217


def powerset(
    iterable: Iterable[Any],
    nonempty: bool = False,
    reverse: bool = False,
    min_size: int = 0,
    max_size: int | None = None,
) -> chain[Any]:
    """Generate the power set of an iterable.

    Parameters
    ----------
    iterable : Iterable
        The iterable of which to generate the power set.

    Other Parameters
    ----------------
    nonempty : bool
        If ``True``, do not include the empty set.
    reverse : bool
        If ``True``, reverse the order of the power set.
    min_size : int
        Only generate subsets of this size or larger (default 0). When greater
        than 0, this supersedes ``nonempty``.
    max_size : int or None
        Only generate subsets of this size or smaller. If ``None`` (the
        default), the maximum is the length of ``iterable``.

    Returns
    -------
    Iterable
        An iterator over the power set.

    Examples
    --------
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

    The files should be stored in ``../data/<directory>`` and named
    ``0.npy, 1.npy, ... <num - 1>.npy``.

    Returns
    -------
    list
        A list of loaded data, such that ``list[i]`` contains the contents of
        ``i.npy``.
    """
    root = Path(__file__).parent.resolve()

    def get_path(i: int) -> Path:  # pylint: disable=missing-docstring
        return root / "data" / directory / f"{i}.npy"

    return [np.load(get_path(i), allow_pickle=True) for i in range(num)]


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


NO_DEFAULT = object()
"""Sentinel distinguishing "no default given" from an explicit ``None``."""


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
