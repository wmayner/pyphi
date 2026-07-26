"""Byte-weighted LRU eviction under the cache memory ceiling."""

import numpy as np
import pytest

from pyphi import config
from pyphi.cache import cache_utils
from pyphi.cache.cache_utils import _ENTRY_OVERHEAD_BYTES
from pyphi.cache.cache_utils import entry_weight
from pyphi.cache.content import ContentCache


class _Carrier:
    """A weakref-able stand-in for a System/Substrate source object."""


@pytest.fixture
def over_ceiling(monkeypatch):
    """Report resident memory as over the cache ceiling."""
    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)


@pytest.fixture
def under_ceiling(monkeypatch):
    """Report resident memory as under the cache ceiling."""
    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)


def _fill(cache, fp, keys):
    for k in keys:
        cache.get_or_compute(fp, (k,), lambda: np.zeros(4))


def test_weight_tracks_admitted_and_evicted_entries(under_ceiling):
    cache = ContentCache("test.weight")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)
    assert cache.nbytes == 0
    value = np.zeros(8)
    cache.get_or_compute(fp, (1,), lambda: value)
    assert cache.nbytes == entry_weight(value)
    cache.evict(fp)
    assert cache.nbytes == 0
    assert cache.size == 0


def test_eviction_fires_under_a_byte_budget(monkeypatch):
    """Past the ceiling the cache stops growing but keeps admitting."""
    cache = ContentCache("test.evicts")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)

    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)
    _fill(cache, fp, range(20))
    bounded_at = cache.nbytes
    assert cache.size == 20

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
    _fill(cache, fp, range(100, 200))

    assert cache.evictions > 0
    assert cache.nbytes <= bounded_at
    # Still admitting: the newest key is present, unlike under a freeze.
    assert (fp, (199,)) in cache._cache


def test_working_set_survives_eviction(monkeypatch):
    """Entries kept warm by reuse outlive entries touched once."""
    cache = ContentCache("test.working_set")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)

    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)
    hot = list(range(5))
    _fill(cache, fp, hot)
    _fill(cache, fp, range(100, 120))

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
    for cold in range(200, 260):
        # Touch the working set, then admit an entry that has to displace
        # something. Recency must spare the reused keys.
        for k in hot:
            cache.get_or_compute(fp, (k,), lambda: pytest.fail("evicted hot key"))
        cache.get_or_compute(fp, (cold,), lambda: np.zeros(4))

    for k in hot:
        assert (fp, (k,)) in cache._cache


def test_hit_rate_stays_nonzero_past_the_budget(monkeypatch):
    """Past the ceiling, repeated access still hits rather than always missing."""
    cache = ContentCache("test.hitrate")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)

    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)
    _fill(cache, fp, range(50))

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
    hits_before = cache.hits
    # A working set well inside the bound, accessed repeatedly.
    for _ in range(10):
        for k in range(10):
            cache.get_or_compute(fp, (k,), lambda: np.zeros(4))
    assert cache.hits - hits_before > 0


def test_entry_larger_than_budget_is_not_admitted(over_ceiling):
    """An entry that cannot fit does not flush the cache to store itself."""
    cache = ContentCache("test.toobig")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)
    # Bound latches at zero: nothing was cached before the ceiling was reached.
    cache.get_or_compute(fp, (1,), lambda: np.zeros(1024))
    assert cache.size == 0
    assert cache.nbytes == 0


def test_bound_lifts_when_memory_is_released(monkeypatch):
    """A bound latched during a spike does not outlive the spike."""
    cache = ContentCache("test.unlatch")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
    cache.get_or_compute(fp, (1,), lambda: np.zeros(4))
    assert cache.size == 0

    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)
    cache.get_or_compute(fp, (2,), lambda: np.zeros(4))
    assert cache.size == 1


def test_weight_counts_payload_not_just_entries(under_ceiling):
    """Byte weight separates a large repertoire from a small one."""
    cache = ContentCache("test.bytes")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)
    cache.get_or_compute(fp, ("small",), lambda: np.zeros(2))
    small = cache.nbytes
    cache.get_or_compute(fp, ("large",), lambda: np.zeros(100_000))
    assert cache.nbytes - small > 100_000 * 8


def test_view_is_charged_without_its_base_buffer():
    """A view's buffer belongs to the array it derives from."""
    base = np.zeros(100_000)
    view = base[:50_000]
    assert view.base is not None
    assert entry_weight(view) < _ENTRY_OVERHEAD_BYTES + 1000
    assert entry_weight(base) > 100_000 * 8


def test_clear_resets_weight_and_bound(over_ceiling):
    cache = ContentCache("test.clear")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)
    cache._store._budget = 10**9
    cache.get_or_compute(fp, (1,), lambda: np.zeros(4))
    assert cache.nbytes > 0
    cache.clear()
    assert cache.nbytes == 0
    assert cache.size == 0
    assert cache.evictions == 0
    assert cache._store._budget is None


def test_info_reports_bytes_and_evictions(under_ceiling):
    from pyphi import cache as cache_module

    cache = ContentCache("test.info_surface")
    fp = b"fp"
    carrier = _Carrier()
    cache.observe(carrier, fp)
    cache.get_or_compute(fp, (1,), lambda: np.zeros(8))
    info = cache_module.info()["test.info_surface"]
    assert info.currsize == 1
    assert info.nbytes == cache.nbytes > 0
    assert info.evictions == 0


def test_module_level_cache_is_byte_bounded(monkeypatch):
    """The @cache decorator evicts rather than freezing at the ceiling.

    ``max_entropy_distribution`` is keyed on a purview, so its entry count
    grows as 2ⁿ; it is the module-level cache that needs a bound.
    """
    from pyphi.distribution import max_entropy_distribution as med

    med.cache_clear()
    nodes = tuple(range(8))
    purviews = [(i,) for i in range(8)] + [(0, 1), (0, 1, 2), (0, 1, 2, 3)]

    monkeypatch.setattr(cache_utils, "memory_full", lambda: False)
    for pv in purviews:
        med(nodes, pv, None)
    grown = med.cache_info()
    assert grown.currsize == len(purviews)
    assert grown.nbytes > 0
    assert grown.evictions == 0

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
    for pv in [(3, 4, 5), (2, 4, 6), (1, 5, 7), (0, 2, 4, 6), (1, 3, 5, 7)]:
        med(nodes, pv, None)
    bounded = med.cache_info()
    assert bounded.evictions > 0
    assert bounded.nbytes <= grown.nbytes
    # Still admitting past the ceiling, unlike a freeze.
    assert med(nodes, (1, 3, 5, 7), None) is not None
    assert med.cache_info().hits > grown.hits
    med.cache_clear()


def test_module_level_cache_weighs_values_not_entries():
    """A big index table outweighs many small distributions."""
    from pyphi.partition import directed_tripartition_indices as dtpi

    dtpi.cache_clear()
    dtpi(3)
    small = dtpi.cache_info().nbytes
    dtpi(9)
    large = dtpi.cache_info().nbytes - small
    assert large > 100 * small
    dtpi.cache_clear()
    assert dtpi.cache_info().nbytes == 0


def test_kernel_cache_admits_past_the_ceiling(monkeypatch):
    """The shard path keeps caching a recent working set past its budget."""
    from pyphi import examples
    from pyphi.core import repertoire_algebra

    repertoire_algebra.clear_caches()
    system = examples.basic_system()
    try:
        with config.override(memory_ceiling_bytes=1 << 60):
            system.cause_repertoire((0,), (1,))
        sizes = {n: c.size for n, c in repertoire_algebra._kernel_caches.items()}
        assert any(size > 0 for size in sizes.values())

        # Now over the ceiling: entries are traded, not refused outright.
        monkeypatch.setattr(cache_utils, "memory_full", lambda: True)
        system.cause_repertoire((0, 1), (0, 1))
        assert any(c.size > 0 for c in repertoire_algebra._kernel_caches.values())
    finally:
        repertoire_algebra.clear_caches()
