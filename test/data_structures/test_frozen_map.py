"""Behaviour of :class:`pyphi.data_structures.FrozenMap`.

The hash tests are the load-bearing ones: ``FrozenMap`` is the key type for
memoized repertoires, so a hash that fails to separate distinct mappings turns
every cache operation into a linear scan.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from types import MappingProxyType

import pytest

from pyphi.data_structures import FrozenMap


def _conditions(n: int) -> list[FrozenMap]:
    """Every binary mechanism state over ``n`` nodes, as a condition mapping.

    This is the population the repertoire cache actually keys on.
    """
    return [
        FrozenMap(dict(enumerate(state)))
        for state in itertools.product((0, 1), repeat=n)
    ]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 10])
def test_hash_separates_every_condition_over_a_mechanism(n):
    """Distinct conditions must not share a hash.

    Hashing the key set and the value set separately satisfies the hash
    contract but collapses all 2ⁿ conditions onto three hashes, since they
    share a key set and draw values from ``{0, 1}``.
    """
    conditions = _conditions(n)
    assert len(set(conditions)) == 2**n
    assert len(set(map(hash, conditions))) == 2**n


def test_hash_depends_on_which_key_holds_which_value():
    assert hash(FrozenMap({0: 1, 1: 0})) != hash(FrozenMap({0: 0, 1: 1}))


def test_hash_is_independent_of_insertion_order():
    assert hash(FrozenMap({1: 2, 3: 4})) == hash(FrozenMap({3: 4, 1: 2}))


def test_equal_maps_are_interchangeable_as_keys():
    a = FrozenMap({1: 2, 3: 4})
    b = FrozenMap({3: 4, 1: 2})
    assert a == b
    assert len({a, b}) == 1
    assert {a: "x"}[b] == "x"


def test_hash_is_computed_once():
    a = FrozenMap({1: 2})
    assert hash(a) == hash(a)
    assert a._hash is not None


@pytest.mark.parametrize(
    "other",
    [{1: 2, 3: 4}, OrderedDict({1: 2, 3: 4}), MappingProxyType({1: 2, 3: 4})],
)
def test_equality_with_other_mapping_types(other):
    """Equality is by content against any mapping, in both directions."""
    a = FrozenMap({1: 2, 3: 4})
    assert a == other
    assert other == a


@pytest.mark.parametrize("other", [[1, 2], None, 3, "ab"])
def test_inequality_with_non_mappings(other):
    assert FrozenMap({1: 2}) != other


def test_mapping_interface():
    a = FrozenMap({1: 2, 3: 4})
    assert len(a) == 2
    assert a[1] == 2
    assert 1 in a
    assert sorted(a) == [1, 3]
    assert sorted(a.items()) == [(1, 2), (3, 4)]
    assert repr(a) == "FrozenMap({1: 2, 3: 4})"


def test_replace_returns_a_new_map():
    a = FrozenMap({"x": 1})
    b = a.replace(x=2)
    assert a["x"] == 1
    assert b["x"] == 2
    assert hash(a) != hash(b)


def test_defining_equality_left_the_type_hashable():
    """Equality and hashing must be defined together.

    Defining ``__eq__`` in a class body sets ``__hash__`` to ``None`` unless
    ``__hash__`` is defined alongside it, which would make instances unusable
    as the cache keys they exist to be.
    """
    assert FrozenMap.__hash__ is not None
    assert isinstance(hash(FrozenMap({1: 2})), int)


def test_equality_with_a_frozen_map_does_not_go_through_getitem():
    """The fast path compares the underlying dicts directly.

    The inherited mapping equality builds a dict from each operand through
    ``__getitem__``, which a cache lookup would pay on every hit.
    """
    calls = []
    a = FrozenMap({1: 2, 3: 4})
    b = FrozenMap({1: 2, 3: 4})
    original = FrozenMap.__getitem__

    def counting(self, key):
        calls.append(key)
        return original(self, key)

    FrozenMap.__getitem__ = counting  # type: ignore[method-assign]
    try:
        assert a == b
    finally:
        FrozenMap.__getitem__ = original  # type: ignore[method-assign]
    assert calls == []
