"""Tests for the process-local cache registry."""

from __future__ import annotations

import pytest

from pyphi.cache import registry as reg
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.policy import _DictCacheAdapter


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the registry around each test."""
    snapshot = dict(reg._registry)
    reg._registry.clear()
    yield
    reg._registry.clear()
    reg._registry.update(snapshot)


def _make_adapter(
    name: str,
    contents: dict | None = None,
    stats: tuple[int, int] = (0, 0),
) -> _DictCacheAdapter:
    backing = contents if contents is not None else {}
    return _DictCacheAdapter(name=name, backing=backing, stats=lambda: stats)


def test_register_and_info_roundtrip():
    adapter = _make_adapter("test.x", {"k": "v"}, stats=(1, 2))
    reg.register(adapter)
    info = reg.info()
    assert "test.x" in info
    assert info["test.x"] == _CacheInfo(1, 2, 1)


def test_clear_one_clears_only_that_cache():
    a = _make_adapter("test.a", {"k1": 1})
    b = _make_adapter("test.b", {"k2": 2})
    reg.register(a)
    reg.register(b)
    reg.clear("test.a")
    assert a.backing == {}
    assert b.backing == {"k2": 2}


def test_clear_all_clears_every_registered_cache():
    a = _make_adapter("test.a", {"k": 1})
    b = _make_adapter("test.b", {"k": 2})
    reg.register(a)
    reg.register(b)
    reg.clear_all()
    assert a.backing == {}
    assert b.backing == {}


def test_unregister_removes_entry():
    a = _make_adapter("test.a")
    reg.register(a)
    reg.unregister("test.a")
    assert "test.a" not in reg.info()


def test_duplicate_registration_replaces_silently():
    """Module reloads / fixture re-registration should not error."""
    a1 = _make_adapter("test.a", {"k1": 1})
    a2 = _make_adapter("test.a", {"k2": 2})
    reg.register(a1)
    reg.register(a2)
    assert reg.info()["test.a"].currsize == 1
    assert a1 not in reg._registry.values()
    assert a2 in reg._registry.values()


def test_clear_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        reg.clear("test.nonexistent")


def test_unregister_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        reg.unregister("test.nonexistent")


def test_pyphi_cache_re_exports_registry_surface():
    """Top-level pyphi.cache exposes info / clear_all / clear / register."""
    from pyphi import cache

    assert callable(cache.info)
    assert callable(cache.clear_all)
    assert callable(cache.clear)
    assert callable(cache.register)
    assert callable(cache.unregister)
