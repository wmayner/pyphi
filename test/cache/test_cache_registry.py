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
    from pyphi import partition  # noqa: F401

    info = cache_module.info()
    expected_name = "pyphi.partition.bipartition_indices"
    assert expected_name in info, (
        f"expected {expected_name} in registry, got keys: {sorted(info.keys())}"
    )


def test_module_level_caches_present_for_partition_and_distribution():
    """Each module that uses @cache(...) shows up under its qualified name."""
    from pyphi import cache as cache_module
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


def test_registry_clear_resets_decorator_store_accounting():
    """Guards defect: ``_DictCacheAdapter.clear()`` emptied the backing dict
    directly, bypassing ``ByteBoundedStore`` accounting — ``info()`` reported
    stale nonzero nbytes with currsize 0, and a latched budget kept refusing
    admissions after the clear even once memory pressure was gone."""
    from pyphi import config
    from pyphi.cache import cache

    @cache()
    def f(i):
        return list(range(100))

    name = f"{f.__module__}.{f.__qualname__}"

    with config.override(memory_ceiling_bytes=10**15):  # ample: fill freely
        for i in range(20):
            f(i)
    with config.override(memory_ceiling_bytes=1):  # ceiling reached: latch
        f(100)

    reg.clear(name)
    info = reg.info()[name]
    assert info.currsize == 0
    assert info.nbytes == 0  # stale weight was the defect

    with config.override(memory_ceiling_bytes=10**15):  # pressure gone
        for i in range(20):
            f(i)
        assert reg.info()[name].currsize == 20  # latched budget was the defect
        hits_before = reg.info()[name].hits
        f(0)
        assert reg.info()[name].hits == hits_before + 1


def test_registry_clear_resets_content_cache_store_accounting():
    """Same defect as above, for the ContentCache registration site."""
    from pyphi import config
    from pyphi.cache.content import ContentCache

    c = ContentCache("test.clear_resets_store")
    with config.override(memory_ceiling_bytes=10**15):
        for i in range(20):
            c.get_or_compute(b"fp", (i,), lambda: list(range(100)))
    with config.override(memory_ceiling_bytes=1):
        c.get_or_compute(b"fp", ("latch",), lambda: list(range(100)))

    reg.clear("test.clear_resets_store")
    info = reg.info()["test.clear_resets_store"]
    assert info.currsize == 0
    assert info.nbytes == 0

    with config.override(memory_ceiling_bytes=10**15):
        for i in range(30):
            c.get_or_compute(b"fp", ("post", i), lambda: list(range(100)))
        assert c.size == 30


def test_substrate_purview_cache_is_a_singleton_registration():
    """The potential-purview cache is one module-level ``ContentCache``
    (registered once, at import, as ``substrate.potential_purviews``), not a
    per-Substrate object — so constructing substrates registers no new keys."""
    from pyphi import cache as cache_module
    from pyphi import examples

    before = set(cache_module.info().keys())
    examples.basic_substrate()
    examples.basic_substrate()
    after = set(cache_module.info().keys())
    assert not (after - before)  # no per-Substrate registrations
    assert "substrate.potential_purviews" in before  # the singleton is registered
