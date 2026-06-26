"""Tests for the CachePolicy Protocol and adapters."""

from __future__ import annotations

from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.policy import CachePolicy
from pyphi.cache.policy import _DictCacheAdapter


def test_dict_cache_adapter_is_a_cache_policy():
    """Concrete adapters must structurally satisfy the Protocol."""
    backing: dict[str, int] = {"a": 1, "b": 2}
    adapter = _DictCacheAdapter(
        name="test.adapter",
        backing=backing,
        stats=lambda: (3, 4),
    )
    p: CachePolicy = adapter
    assert p.name == "test.adapter"
    info = p.info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 3
    assert info.misses == 4
    assert info.currsize == 2


def test_dict_cache_adapter_clear_clears_backing():
    backing: dict[str, int] = {"a": 1}
    adapter = _DictCacheAdapter(
        name="test.adapter",
        backing=backing,
        stats=lambda: (0, 0),
    )
    adapter.clear()
    assert backing == {}


def test_dict_cache_adapter_stats_callable_is_invoked_each_call():
    """info() should call the stats callable fresh, not snapshot at construction."""
    counts = [0, 0]

    def stats() -> tuple[int, int]:
        return (counts[0], counts[1])

    adapter = _DictCacheAdapter(
        name="test.live_stats",
        backing={},
        stats=stats,
    )
    counts[0] = 5
    counts[1] = 7
    info = adapter.info()
    assert info.hits == 5
    assert info.misses == 7
