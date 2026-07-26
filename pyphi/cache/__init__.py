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

from functools import update_wrapper

import joblib

from pyphi import constants

from .cache_utils import ByteBoundedStore
from .cache_utils import _CacheInfo
from .cache_utils import _make_key

# An on-disk cache for distributing pre-computed results with the PyPhi package
joblib_memory = joblib.Memory(location=constants.DISK_CACHE_LOCATION, verbose=0)


def cache(typed: bool = False):
    """Memoization decorator bounded by bytes.

    Arguments to the cached function must be hashable. Entries are held in a
    :class:`pyphi.cache.cache_utils.ByteBoundedStore`, so the cache grows
    freely until resident memory reaches the configured ceiling and then holds
    its occupancy steady, evicting least recently used entries to admit new
    ones.

    The bound is on bytes rather than on entry count because a cached value
    may be anything from a scalar to an array whose size grows exponentially
    in the size of the system.

    Parameters
    ----------
    typed : bool, optional
        If ``True``, arguments of different types are cached separately: for
        example, ``f(3.0)`` and ``f(3)`` are treated as distinct calls with
        distinct results. Defaults to ``False``.

    Notes
    -----
    The decorated function exposes ``cache_info()``, which returns a
    ``(hits, misses, currsize, nbytes, evictions)`` named tuple;
    ``cache_clear()``, which empties the cache and resets its statistics; and
    ``__wrapped__``, the underlying function.
    """
    store = ByteBoundedStore()
    entries = store.data
    # Unique object used to signal cache misses.
    sentinel = object()
    # Build a key from the function arguments.
    make_key = _make_key

    def decorating_function(user_function, hits=0, misses=0):
        # Bound method to look up a key or return None.
        cache_get = entries.get

        def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                # Reinsert to move the entry to the recent end of the store's
                # iteration order, which is the eviction order.
                del entries[key]
                entries[key] = result
                return result
            result = user_function(*args, **kwds)
            store.admit(key, result)
            misses += 1
            return result

        def cache_info():
            """Report cache statistics."""
            return _CacheInfo(hits, misses, len(entries), store.nbytes, store.evictions)

        def cache_clear():
            """Clear the cache and cache statistics."""
            nonlocal hits, misses
            store.clear()
            hits = misses = 0

        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        # Register a CachePolicy adapter under '<module>.<qualname>'.
        from .policy import _DictCacheAdapter
        from .registry import register as _register_policy

        _register_policy(
            _DictCacheAdapter(
                name=f"{user_function.__module__}.{user_function.__qualname__}",
                backing=entries,
                stats=lambda: (hits, misses),
                weigh=lambda: (store.nbytes, store.evictions),
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
