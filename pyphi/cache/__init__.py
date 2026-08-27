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

The caches built by the ``cache`` decorator below are module-level, so the
thread scheduler shares them across worker threads. They follow the same
design as ``ContentCache``: the hit path is lock-free, using only atomic dict
operations, and admission (which runs the store's eviction loop) is locked.
Their counters carry the same best-effort caveat under free-threading.

Public surface
--------------
- ``info()``: dict of name -> _CacheInfo across every registered cache.
- ``clear_all()``: clear every registered in-memory cache. Persistent
  on-disk stores are skipped; clear those by name.
- ``clear(name)``: clear one cache by name (including persistent ones).
- ``register(policy)``: register a CachePolicy adapter.
- ``unregister(name)``: remove a registration.

See :mod:`pyphi.cache.policy` for the CachePolicy Protocol and
:mod:`pyphi.cache.registry` for the registry implementation.
"""

import threading
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
    # Guards admission and clearing, which mutate the store's weight
    # bookkeeping; the hit path is lock-free (atomic dict operations only),
    # since the thread scheduler shares these caches across worker threads.
    lock = threading.Lock()

    def decorating_function(user_function, hits=0, misses=0):
        # Bound method to pop a key or return the sentinel.
        cache_pop = entries.pop

        def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            # Atomic pop, then reinsert, to move the entry to the recent end
            # of the store's iteration order, which is the eviction order.
            # Unlike get-then-delete, the pop cannot raise when another
            # thread hits the same key concurrently.
            result = cache_pop(key, sentinel)
            if result is not sentinel:
                hits += 1
                entries[key] = result
                return result
            result = user_function(*args, **kwds)
            with lock:
                store.admit(key, result)
            misses += 1
            return result

        def cache_info():
            """Report cache statistics."""
            return _CacheInfo(hits, misses, len(entries), store.nbytes, store.evictions)

        def cache_clear():
            """Clear the cache and cache statistics."""
            nonlocal hits, misses
            with lock:
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
                reset=cache_clear,
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
