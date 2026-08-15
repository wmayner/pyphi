"""Hash quality of every type PyPhi uses as a cache key.

A hash that satisfies the equality contract can still be useless: if distinct
keys collide, every dictionary operation on them degenerates to a linear scan
compared under ``__eq__``, and a cache holding m entries costs O(m²) to fill.
The cost is invisible to correctness tests and to call-count regression gates,
because neither the values nor the number of PyPhi-level operations change.

Two tests guard this. :func:`test_hash_separates_distinct_keys` checks the
declared key types against a generated population. :func:`test_registry_covers_
every_observed_key_type` runs real analyses with the cache instrumented and
fails if a key type appears that the registry does not declare, so a new key
type cannot enter the cache without a hash-quality check coming with it.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence

import pytest

from pyphi import actual
from pyphi import examples
from pyphi.cache.content import ContentCache
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.data_structures import FrozenMap
from pyphi.direction import Direction
from pyphi.system import System

# Types whose hash is Python's own and needs no check here.
TRUSTED = (bytes, str, int, float, bool, type(None), tuple, frozenset, enum.Enum)

# Every PyPhi-defined type that reaches a cache key, mapped to a generator of
# a population of distinct instances. Adding a key type means adding an entry;
# ``test_registry_covers_every_observed_key_type`` enforces that.
KEY_TYPES: dict[type, Callable[[], Sequence[object]]] = {
    FrozenMap: lambda: [
        FrozenMap(dict(enumerate(state)))
        for state in itertools.product((0, 1), repeat=8)
    ],
    Direction: lambda: list(Direction),
}


@pytest.mark.parametrize("key_type", list(KEY_TYPES), ids=lambda t: t.__name__)
def test_hash_separates_distinct_keys(key_type):
    population = KEY_TYPES[key_type]()
    distinct_values = len(set(population))
    distinct_hashes = len(set(map(hash, population)))
    assert distinct_hashes == distinct_values, (
        f"{key_type.__name__} maps {distinct_values} distinct keys onto "
        f"{distinct_hashes} hashes; cache operations on it are quadratic"
    )


def _key_component_types(key: object, into: set[type]) -> None:
    """Collect the types of a cache key's leaves, descending containers."""
    into.add(type(key))
    if isinstance(key, Mapping):
        for k, v in key.items():
            _key_component_types(k, into)
            _key_component_types(v, into)
    elif isinstance(key, (tuple, list, frozenset, set)):
        for item in key:
            _key_component_types(item, into)


def _observed_key_types(run: Callable[[], object]) -> set[type]:
    """Run ``run`` with the content cache instrumented; return key types seen."""
    observed: set[type] = set()
    original = ContentCache.get_or_compute

    def instrumented(self, fingerprint, args, compute, *, store=True):
        _key_component_types(fingerprint, observed)
        _key_component_types(args, observed)
        return original(self, fingerprint, args, compute, store=store)

    ContentCache.get_or_compute = instrumented
    try:
        run()
    finally:
        ContentCache.get_or_compute = original
    return observed


def _iit4_structure() -> None:
    with config.override(**presets.iit4_2023):
        System(
            substrate=examples.basic_substrate(),
            state=(1, 0, 0),
            node_indices=(0, 1, 2),
        ).ces()


def _iit3_structure() -> None:
    with config.override(**presets.iit3):
        System(
            substrate=examples.basic_substrate(),
            state=(1, 0, 0),
            node_indices=(0, 1, 2),
        ).ces()


def _actual_causation() -> None:
    actual.account(examples.prevention_transition())


WORKLOADS = {
    "iit4_structure": _iit4_structure,
    "iit3_structure": _iit3_structure,
    "actual_causation": _actual_causation,
}


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_registry_covers_every_observed_key_type(name):
    """No key type may reach the cache without a declared hash-quality check.

    Failing here means a new type became part of a cache key. Add it to
    ``KEY_TYPES`` with a generator producing a population of distinct
    instances, or to ``TRUSTED`` if its hash is Python's own.
    """
    observed = _observed_key_types(WORKLOADS[name])
    undeclared = {
        t for t in observed if not issubclass(t, TRUSTED) and t not in KEY_TYPES
    }
    assert not undeclared, (
        "cache-key types with no declared hash-quality check: "
        f"{sorted(t.__module__ + '.' + t.__name__ for t in undeclared)}"
    )


def test_the_instrumentation_actually_sees_the_key_types():
    """Guard the guard: an instrumentation that observes nothing passes vacuously."""
    observed = _observed_key_types(_iit4_structure)
    assert FrozenMap in observed
    assert Direction in observed
