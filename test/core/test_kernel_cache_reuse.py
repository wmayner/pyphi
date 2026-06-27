from pyphi import examples
from pyphi.core import repertoire_algebra as ra


def test_distinct_equivalent_systems_share_kernel_cache():
    ra.clear_caches()
    s1 = examples.basic_system()
    s2 = examples.basic_system()  # distinct object, same math
    phi1 = s1.sia().phi
    sizes_after_first = sum(c["size"] for c in ra.cache_info().values())
    assert sizes_after_first > 0
    phi2 = s2.sia().phi  # should hit the cache populated by s1
    assert phi1 == phi2
    # No new kernel entries created for the equivalent second system.
    sizes_after_second = sum(c["size"] for c in ra.cache_info().values())
    assert sizes_after_second == sizes_after_first
