import pickle
from collections.abc import Hashable
from functools import partial
from itertools import tee as _tee

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.strategies import composite

from pyphi.network import Network
from pyphi.subsystem import Subsystem


class PrettyIter:
    """An iterator that displays its contents."""

    def __init__(self, values):
        self._values, _repr = _tee(values, 2)
        self._repr = list(_repr)
        self._iter = iter(self._values)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return f"iter({self._repr!r})"


def everything_except(*excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def anything():
    return everything_except()


def finite_floats(min_value=None, max_value=None):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@composite
def anything_comparable(draw):
    example = draw(anything())
    try:
        assume(example == example)  # noqa: PLR0124
    except Exception:
        assume(False)
    return example


@composite
def anything_pickleable(draw):
    example = draw(anything())
    try:
        assume(example == pickle.loads(pickle.dumps(example)))
    except Exception:
        assume(False)
    return example


@composite
def anything_pickleable_and_hashable(draw):
    example = draw(anything())
    try:
        assume(
            isinstance(example, Hashable)
            and example == pickle.loads(pickle.dumps(example))
        )
    except Exception:
        assume(False)
    return example


@composite
def list_and_index(draw, elements):
    items = draw(st.lists(elements))
    n = len(items)
    index = draw(st.integers(min_value=0, max_value=(n - 1) if n else 0))
    return (items, index)


def iterable_or_list(elements):
    return st.iterables(elements) | st.iterables(elements)


def tee(iterable, n=2):
    return tuple(map(PrettyIter, _tee(iterable, n)))


def teed(strategy, n=2):
    return strategy.map(partial(tee, n=n))


# ============================================================================
# PyPhi-specific strategies for property-based invariant testing
# ============================================================================
#
# These strategies generate small binary networks (2-3 nodes by default) for
# Hypothesis property tests. They deliberately avoid invoking subsystem state
# validation: random TPMs frequently produce states with zero past probability,
# which is fine mathematically but trips ``validate.state_reachable``. Tests
# using these strategies should run inside
# ``config.override(VALIDATE_SUBSYSTEM_STATES=False)``.


def binary_state(n):
    """Strategy for a binary state of length ``n`` as a tuple of 0/1 ints."""
    return st.tuples(*([st.integers(0, 1)] * n))


@composite
def binary_state_by_node_tpm(draw, n, deterministic=False):
    """Strategy for a state-by-node TPM with ``n`` binary nodes.

    Shape ``(2**n, n)`` with values in ``[0, 1]``. When ``deterministic`` is
    True, values are exactly 0 or 1 (useful for "causally perfect" tests).
    """
    rows = 2**n
    if deterministic:
        bits = draw(
            st.lists(
                st.integers(0, 1),
                min_size=rows * n,
                max_size=rows * n,
            )
        )
        arr = np.array(bits, dtype=np.float64).reshape(rows, n)
    else:
        # Discretize to a few levels so Hypothesis can shrink usefully and we
        # don't hit pathological floats.
        level = draw(
            st.lists(
                st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
                min_size=rows * n,
                max_size=rows * n,
            )
        )
        arr = np.array(level, dtype=np.float64).reshape(rows, n)
    return arr


@composite
def fully_connected_cm(draw, n):
    """Strategy for an all-ones connectivity matrix of size ``n``."""
    _ = draw(st.just(0))  # consume one draw to satisfy composite
    return np.ones((n, n), dtype=int)


@composite
def random_cm(draw, n):
    """Strategy for a random binary connectivity matrix of size ``n``.

    Self-loops (the diagonal) are always included; off-diagonal entries are
    drawn independently. This avoids fully-disconnected nodes that produce
    trivial subsystems.
    """
    bits = draw(st.lists(st.integers(0, 1), min_size=n * n, max_size=n * n))
    cm = np.array(bits, dtype=int).reshape(n, n)
    np.fill_diagonal(cm, 1)
    return cm


@composite
def small_network(
    draw,
    min_size=2,
    max_size=3,
    deterministic=False,
    fully_connected=True,
):
    """Strategy for a small ``Network``.

    Args:
        min_size, max_size: number of nodes (defaults give 2-3 node nets so
            Hypothesis can explore many seeds in reasonable time).
        deterministic: if True, every TPM entry is exactly 0 or 1.
        fully_connected: if True, use an all-ones CM; else draw a random CM
            with self-loops guaranteed.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    tpm = draw(binary_state_by_node_tpm(n, deterministic=deterministic))
    cm = draw(fully_connected_cm(n) if fully_connected else random_cm(n))
    return Network(tpm, cm=cm)


@composite
def small_subsystem(
    draw,
    min_size=2,
    max_size=3,
    deterministic=False,
    fully_connected=True,
):
    """Strategy for a ``Subsystem`` over the full network.

    Random TPMs frequently produce states whose backward TPM has zero
    normalization (``StateUnreachableBackwardsError``). Those states are
    discarded via ``assume`` rather than constructed-then-raised, so callers
    don't need to handle the exception.
    """
    from pyphi.exceptions import StateUnreachableBackwardsError

    network = draw(
        small_network(
            min_size=min_size,
            max_size=max_size,
            deterministic=deterministic,
            fully_connected=fully_connected,
        )
    )
    state = draw(binary_state(network.size))
    try:
        return Subsystem(network, state, network.node_indices)
    except StateUnreachableBackwardsError:
        assume(False)
        # Unreachable; assume(False) raises UnsatisfiedAssumption.
        raise


@composite
def mechanism_purview_pair(draw, subsystem, allow_empty=False):
    """Strategy for a ``(mechanism, purview)`` pair drawn from a subsystem.

    Both default to nonempty (the typical case for repertoire computation);
    set ``allow_empty=True`` to allow the empty mechanism, which yields the
    unconstrained repertoire by definition.
    """
    nodes = list(subsystem.node_indices)
    min_mech = 0 if allow_empty else 1
    mechanism = tuple(
        sorted(draw(st.lists(st.sampled_from(nodes), min_size=min_mech, unique=True)))
    )
    purview = tuple(
        sorted(draw(st.lists(st.sampled_from(nodes), min_size=1, unique=True)))
    )
    return mechanism, purview
