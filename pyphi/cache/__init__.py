# cache/__init__.py
"""Memoization and caching utilities.

Threading
---------
``ContentCache`` (see :mod:`pyphi.cache.content`) is safe for concurrent use
by worker threads: cached values are correct, eviction is sound, and no
operation raises under concurrent access. Its ``hits``/``misses`` counters are
best-effort under free-threaded Python — exact under the GIL and under
process-isolated parallelism, approximate when threads share one cache — since
they are diagnostics that nothing computes on, and are deliberately left out of
the lock to keep the hot path free of contention.

The ``cache`` decorator below is oriented to process-isolated parallelism
(each worker process owns its caches) and is not shared across threads by the
current schedulers; its counters carry the same best-effort caveat under
free-threading.

Public surface
--------------
- ``info()``: dict of name -> _CacheInfo across every registered cache.
- ``clear_all()``: clear every registered cache.
- ``clear(name)``: clear one cache by name.
- ``register(policy)``: register a CachePolicy adapter.
- ``unregister(name)``: remove a registration.

See :mod:`pyphi.cache.policy` for the CachePolicy Protocol and
:mod:`pyphi.cache.registry` for the registry implementation.
"""

import os
from functools import update_wrapper

import joblib
import psutil

from pyphi import constants
from pyphi.conf import config

from .cache_utils import _CacheInfo
from .cache_utils import _make_key

# An on-disk cache for distributing pre-computed results with the PyPhi package
joblib_memory = joblib.Memory(location=constants.DISK_CACHE_LOCATION, verbose=0)


def cache(
    cache=None,
    maxmem: int | None = config.infrastructure.maximum_cache_memory_percentage,
    typed: bool = False,
):
    """Memory-limited memoization decorator.

    Arguments to the cached function must be hashable.

    Parameters
    ----------
    cache : dict, optional
        Backing store for cached results. A fresh empty dict is created when
        omitted; passing one shares the store across decorated functions.
    maxmem : float or None, optional
        Maximum percentage of physical memory the cache may use, between 0 and
        100 inclusive. ``None`` (or 0) means unlimited. Once the process
        exceeds this fraction of memory, no new entries are stored, though
        entries already cached are still served.
    typed : bool, optional
        If ``True``, arguments of different types are cached separately: for
        example, ``f(3.0)`` and ``f(3)`` are treated as distinct calls with
        distinct results. Defaults to ``False``.

    Notes
    -----
    The decorated function exposes ``cache_info()``, which returns a
    ``(hits, misses, currsize)`` named tuple; ``cache_clear()``, which empties
    the cache and resets its statistics; and ``__wrapped__``, the underlying
    function.
    """
    # Constants shared by all lru cache instances:
    # Unique object used to signal cache misses.
    if cache is None:
        cache = {}
    sentinel = object()
    # Build a key from the function arguments.
    make_key = _make_key

    def decorating_function(user_function, hits=0, misses=0):
        full = False
        # Bound method to look up a key or return None.
        cache_get = cache.get

        if not maxmem:

            def wrapper(*args, **kwds):
                # Simple caching without memory limit.
                nonlocal hits, misses
                key = make_key(args, kwds, typed)
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    hits += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                misses += 1
                return result

        else:
            # Type narrowing: maxmem is not None in this branch
            assert maxmem is not None, "maxmem should not be None in else branch"
            maxmem_value = maxmem

            def wrapper(*args, **kwds):
                # Memory-limited caching.
                nonlocal hits, misses, full
                key = make_key(args, kwds, typed)
                result = cache_get(key)
                if result is not None:
                    hits += 1
                    return result
                result = user_function(*args, **kwds)
                if not full:
                    cache[key] = result
                    # Cache is full if the total recursive usage is greater
                    # than the maximum allowed percentage.
                    current_process = psutil.Process(os.getpid())
                    full = current_process.memory_percent() > maxmem_value
                misses += 1
                return result

        def cache_info():
            """Report cache statistics."""
            return _CacheInfo(hits, misses, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics."""
            nonlocal hits, misses, full
            cache.clear()
            hits = misses = 0
            full = False

        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        # Register a CachePolicy adapter under '<module>.<qualname>'.
        from .policy import _DictCacheAdapter
        from .registry import register as _register_policy

        _register_policy(
            _DictCacheAdapter(
                name=f"{user_function.__module__}.{user_function.__qualname__}",
                backing=cache,
                stats=lambda: (cache_info().hits, cache_info().misses),
            )
        )

        return update_wrapper(wrapper, user_function)

    return decorating_function


# Public registry surface — re-exports placed at the bottom of the module so
# the decorator above is defined before the submodules it shares helpers with
# are imported.
from .content import ContentCache as ContentCache  # noqa: E402
from .disk import DiskCache as DiskCache  # noqa: E402
from .registry import clear as clear  # noqa: E402
from .registry import clear_all as clear_all  # noqa: E402
from .registry import info as info  # noqa: E402
from .registry import register as register  # noqa: E402
from .registry import unregister as unregister  # noqa: E402
