"""Tests for pyphi.core.repertoire_algebra — stateless repertoire functions + cache."""

from __future__ import annotations

import gc

import pytest


def test_memoize_caches_results() -> None:
    """A memoized function returns the cached value on second call."""
    from pyphi.core.repertoire_algebra import _memoize

    call_count = {"n": 0}

    @_memoize
    def f(cs, x):
        call_count["n"] += 1
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    assert f(cs, 3) == 6
    assert f(cs, 3) == 6
    assert call_count["n"] == 1


def test_memoize_evicts_on_gc() -> None:
    """When a System is GC'd, its cache entries are evicted."""
    from pyphi.core.repertoire_algebra import _caches
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    cs_id = id(cs)
    f(cs, 1)
    f(cs, 2)
    assert any(k[0] == cs_id for k in _caches[f.__name__])
    del cs
    gc.collect()
    assert not any(k[0] == cs_id for k in _caches[f.__name__])


def test_memoize_does_not_poison_on_failure() -> None:
    """A raised exception must not pollute the cache."""
    from pyphi.core.repertoire_algebra import _memoize

    call_count = {"n": 0}

    @_memoize
    def f(cs, x):
        call_count["n"] += 1
        if x == 1:
            raise ValueError("boom")
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    with pytest.raises(ValueError):
        f(cs, 1)
    assert f(cs, 2) == 4
    assert call_count["n"] == 2


@pytest.fixture
def cs():
    from pyphi import examples
    from pyphi.system import System

    return System(
        substrate=examples.basic_substrate(),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )


def test_kernel_caches_appear_in_registry() -> None:
    """Each kernel-memoized function registers a policy under kernel.<name>."""
    from pyphi import cache as cache_module
    from pyphi.cache import registry as reg
    from pyphi.core import repertoire_algebra as ra  # noqa: F401  # trigger decoration

    keys = list(reg._registry.keys())
    kernel_keys = [k for k in keys if k.startswith("kernel.")]
    assert kernel_keys, f"expected kernel.* entries; got: {keys}"

    info = cache_module.info()
    assert all(k in info for k in kernel_keys)


def test_kernel_clear_via_registry_clears_kernel_cache(cs) -> None:
    """pyphi.cache.clear('kernel.<name>') empties that kernel cache."""
    from pyphi import cache as cache_module
    from pyphi.core import repertoire_algebra as ra

    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    name = "kernel._single_node_cause_repertoire"
    assert cache_module.info()[name].currsize >= 1

    cache_module.clear(name)
    assert cache_module.info()[name].currsize == 0


def test_kernel_cache_respects_memory_full(monkeypatch, cs) -> None:
    """When memory_full() returns True, kernel cache stops adding entries."""
    from pyphi import cache as cache_module
    from pyphi.cache import cache_utils
    from pyphi.core import repertoire_algebra as ra

    cache_module.clear_all()

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)

    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    ra._single_node_cause_repertoire(cs, 0, frozenset({1, 2}))

    info = cache_module.info()["kernel._single_node_cause_repertoire"]
    assert info.currsize == 0, (
        f"expected 0 cached entries when memory full, got {info.currsize}"
    )
    assert info.misses >= 2
