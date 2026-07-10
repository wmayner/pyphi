Caching
~~~~~~~

PyPhi memoizes expensive computations (repertoires, partition enumerations,
Hamming matrices, ...) through a uniform process-local cache surface in
:mod:`pyphi.cache`:

- ``pyphi.cache.info()``: per-cache statistics (hits, misses, size).
- ``pyphi.cache.clear_all()``: clear every registered cache.
- ``pyphi.cache.clear(name)``: clear one named cache.

The total memory footprint of in-memory caches is bounded by the
``config.infrastructure.maximum_cache_memory_percentage`` option.

Setting ``config.infrastructure.disk_cache_results = True`` additionally
persists whole SIA and cause-effect-structure results to a
``__pyphi_cache__/`` directory, so a repeated analysis of the same system is
served from disk across processes and sessions.

**Note:** the in-memory caches are process-local; each worker in a
process-isolated parallel run has its own copy of every cache.

See the ``Cache results`` how-to guide for worked examples.
