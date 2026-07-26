from pyphi import Direction
from pyphi import config


def test_memory_full_honors_absolute_budget():
    """An absolute budget bounds caching; unset, the percentage still rules."""
    from pyphi.cache import cache_utils

    with config.override(memory_ceiling_bytes=1):
        assert cache_utils.memory_full()
    with config.override(memory_ceiling_bytes=1 << 60):
        assert not cache_utils.memory_full()
    with config.override(memory_ceiling_percentage=100):
        assert not cache_utils.memory_full()
    with config.override(memory_ceiling_percentage=0):
        assert cache_utils.memory_full()


def test_kernel_cache_stops_storing_over_budget():
    """A budget below current usage stops new entries, growth and all."""
    from pyphi import examples
    from pyphi.core import repertoire_algebra

    repertoire_algebra.clear_caches()
    system = examples.basic_system()
    try:
        with config.override(memory_ceiling_bytes=1):
            system.cause_repertoire((0,), (1,))
            assert all(c.size == 0 for c in repertoire_algebra._kernel_caches.values())
        with config.override(memory_ceiling_bytes=1 << 60):
            system.cause_repertoire((0,), (1,))
            assert any(c.size > 0 for c in repertoire_algebra._kernel_caches.values())
    finally:
        repertoire_algebra.clear_caches()


def test_cache_repertoires_config_option():
    """The option gates whether the kernel cache stores repertoires."""
    from pyphi import examples
    from pyphi.core import repertoire_algebra

    repertoire_algebra.clear_caches()
    # Hold the systems alive: kernel cache entries are evicted as soon as
    # their last live carrier is garbage-collected.
    system = examples.basic_system()
    try:
        with config.override(cache_repertoires=False):
            system.cause_repertoire((0,), (1,))
            sizes = {n: c.size for n, c in repertoire_algebra._kernel_caches.items()}
            assert all(size == 0 for size in sizes.values()), sizes
        with config.override(cache_repertoires=True):
            system.cause_repertoire((0,), (1,))
            assert any(c.size > 0 for c in repertoire_algebra._kernel_caches.values())
    finally:
        repertoire_algebra.clear_caches()


# Test purview cache
# ==================


@config.override(cache_potential_purviews=True)
def test_purview_cache(standard):
    from pyphi.substrate import _PURVIEW_CACHE

    _PURVIEW_CACHE.clear()
    purviews = standard.potential_purviews(Direction.EFFECT, (0,))
    assert _PURVIEW_CACHE.size == 1
    again = standard.potential_purviews(Direction.EFFECT, (0,))
    assert again == purviews
    assert _PURVIEW_CACHE.hits >= 1


@config.override(cache_potential_purviews=False)
def test_only_cache_purviews_if_configured(standard):
    from pyphi.substrate import _PURVIEW_CACHE

    _PURVIEW_CACHE.clear()
    standard.potential_purviews(Direction.CAUSE, (0,))
    assert _PURVIEW_CACHE.size == 0  # caching disabled by config
