"""Tests for the process-local cache registry."""

from __future__ import annotations

import pytest

from pyphi.cache import registry as reg
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.policy import _DictCacheAdapter


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the registry at test entry; restore at test exit.

    Does NOT clear at entry — module-level registrations from imports
    (e.g. ``pyphi.partition`` decorators) should remain visible to
    tests that assert on them. Unit tests that need a clean slate can
    use unique ``test.*`` names that won't collide.
    """
    snapshot = dict(reg._registry)
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


# =============================================================================
# Module-level @cache(...) decorator registers an adapter
# =============================================================================


def test_module_level_cache_decorator_registers_adapter():
    """A function decorated with @cache(...) registers a policy under
    f'{module}.{qualname}' on import."""
    from pyphi import cache as cache_module
    from pyphi import combinatorics  # noqa: F401

    info = cache_module.info()
    expected_name = "pyphi.combinatorics.num_subsets_larger_than_one_element"
    assert expected_name in info, (
        f"expected {expected_name} in registry, got keys: {sorted(info.keys())}"
    )


def test_module_level_caches_present_for_partition_and_distribution():
    """Each module that uses @cache(...) shows up under its qualified name."""
    from pyphi import cache as cache_module
    from pyphi import combinatorics  # noqa: F401
    from pyphi import distribution  # noqa: F401
    from pyphi import partition  # noqa: F401

    info = cache_module.info()
    keys = list(info.keys())
    assert any(k.startswith("pyphi.partition.") for k in keys), (
        f"no pyphi.partition.* entries; got: {sorted(keys)}"
    )
    assert any(k.startswith("pyphi.distribution.") for k in keys), (
        f"no pyphi.distribution.* entries; got: {sorted(keys)}"
    )
    assert any(k.startswith("pyphi.combinatorics.") for k in keys), (
        f"no pyphi.combinatorics.* entries; got: {sorted(keys)}"
    )
