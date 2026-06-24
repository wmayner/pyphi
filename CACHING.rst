Caching
~~~~~~~

PyPhi memoizes expensive computations (repertoires, partition enumerations,
Hamming matrices, ...) through a uniform process-local cache surface in
:mod:`pyphi.cache`:

- ``pyphi.cache.info()``: per-cache statistics (hits, misses, size).
- ``pyphi.cache.clear_all()``: clear every registered cache.
- ``pyphi.cache.clear(name)``: clear one named cache.

The total memory footprint of in-memory caches is bounded by the
``MAXIMUM_CACHE_MEMORY_PERCENTAGE`` configuration option.

**Note:** caches are not thread-safe. PyPhi assumes process-isolated
parallelism (Ray-based); each worker has its own copy of every cache.


.. |phi| unicode:: U+1D6BD .. mathematical bold capital phi
