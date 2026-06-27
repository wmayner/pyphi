import gc

from pyphi.cache.content import ContentCache


class _Carrier:
    """A weakref-able stand-in for a System/Substrate source object."""


def test_reuse_across_distinct_objects_with_same_fingerprint():
    cache = ContentCache("test.reuse")
    calls = []

    def compute():
        calls.append(1)
        return "value"

    a, b = _Carrier(), _Carrier()  # distinct objects, shared fingerprint
    fp = b"fp-shared"
    cache.observe(a, fp)
    cache.observe(b, fp)
    r1 = cache.get_or_compute(fp, (1,), compute)
    r2 = cache.get_or_compute(fp, (1,), compute)
    assert r1 == r2 == "value"
    assert len(calls) == 1  # computed once, reused
    assert cache.hits == 1 and cache.misses == 1


def test_refcount_eviction_only_when_last_carrier_dies():
    cache = ContentCache("test.evict")
    fp = b"fp-x"

    class Box:
        pass

    a, b = Box(), Box()
    cache.observe(a, fp)
    cache.observe(b, fp)
    cache.get_or_compute(fp, (), lambda: 42)
    assert cache.size == 1

    del a
    gc.collect()
    assert cache.size == 1  # b still alive -> entry survives

    del b
    gc.collect()
    assert cache.size == 0  # last carrier gone -> purged


def test_store_false_returns_value_without_caching():
    cache = ContentCache("test.nostore")
    fp = b"fp-y"
    r = cache.get_or_compute(fp, (), lambda: 7, store=False)
    assert r == 7
    assert cache.size == 0


def test_evict_purges_only_that_fingerprint():
    cache = ContentCache("test.evict-one")
    cache.get_or_compute(b"fp-a", (), lambda: 1)
    cache.get_or_compute(b"fp-b", (), lambda: 2)
    cache.evict(b"fp-a")
    assert cache.size == 1
    assert cache.get_or_compute(b"fp-b", (), lambda: 99) == 2  # still cached
