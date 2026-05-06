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
    """When a CandidateSystem is GC'd, its cache entries are evicted."""
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
    """If the wrapped function raises, the cache must not retain a partial entry."""
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        if x < 0:
            raise ValueError("negative")
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    with pytest.raises(ValueError):
        f(cs, -1)
    assert f(cs, 4) == 8
    assert f(cs, 4) == 8
