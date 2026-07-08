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

The ``cache`` decorator and ``DictCache`` below are oriented to process-isolated
parallelism (each worker process owns its caches) and are not shared across
threads by the current schedulers; their counters carry the same best-effort
caveat under free-threading.

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
from functools import wraps

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


class DictCache:
    """A generic dictionary-based cache.

    Intended to be used as an object-level cache of method results. If
    ``name`` is provided, the cache registers itself with the cache
    registry on construction; anonymous instances stay out of the
    registry.
    """

    def __init__(self, name: str | None = None):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.name = name
        if name is not None:
            from .policy import _DictCacheAdapter
            from .registry import register as _register_policy

            _register_policy(
                _DictCacheAdapter(
                    name=name,
                    backing=self.cache,
                    stats=lambda: (self.hits, self.misses),
                )
            )

    def clear(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def size(self):
        """Number of items in cache"""
        return len(self.cache)

    def info(self):
        """Return info about cache hits, misses, and size"""
        return _CacheInfo(self.hits, self.misses, self.size())

    def get(self, key):
        """Get a value out of the cache.

        Returns None if the key is not in the cache. Updates cache
        statistics.
        """
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key, value):
        """Set a value in the cache"""
        self.cache[key] = value

    # TODO: handle **kwarg keys if needed
    # See joblib.func_inspect.filter_args
    def key(self, *args, _prefix=None, **kwargs):
        """Build the cache key for the given function arguments.

        The key is the tuple ``(_prefix, *args)``.

        Parameters
        ----------
        *args
            Positional arguments to the cached method, used as the tail of
            the key.
        _prefix : optional
            A constant placed at the head of the key. Defaults to ``None``.

        Raises
        ------
        NotImplementedError
            If any keyword arguments are passed; keyword-argument keys are
            not supported.
        """
        if kwargs:
            raise NotImplementedError("kwarg cache keys not implemented")
        return (_prefix, *tuple(args))


def validate_parent_cache(parent_cache):
    # TODO: also validate that system is a cut version of
    # parent_cache.system? Do we need to check this at all?
    if parent_cache.system.is_partitioned:
        raise ValueError("parent_cache must be from an unpartitioned system")


def method(cache_name, key_prefix=None):
    """Caching decorator for object-level method caches.

    Key generation is delegated to the named cache's ``key`` method, and the
    result is stored via its ``set`` method on a miss. A returned value of
    ``None`` from ``cache.get`` is treated as a miss.

    Parameters
    ----------
    cache_name : str
        Name of an attribute on the decorated object holding an
        already-instantiated cache, used to store results of this method.
    key_prefix : optional
        A constant folded into the cache key alongside the method arguments.
        Defaults to ``None``.

    Notes
    -----
    If ``config.infrastructure.cache_repertoires`` is false and the wrapped
    function's name contains ``"repertoire"``, the decorator returns the
    function unwrapped, so repertoire computations are not cached.
    """

    def decorator(func):
        if not config.infrastructure.cache_repertoires and "repertoire" in func.__name__:
            return func

        @wraps(func)
        def wrapper(obj, *args, **kwargs):
            cache = getattr(obj, cache_name)

            # Delegate key generation
            key = cache.key(*args, _prefix=key_prefix, **kwargs)

            # Get cached value, or compute
            value = cache.get(key)
            if value is None:  # miss
                value = func(obj, *args, **kwargs)
                cache.set(key, value)
            return value

        return wrapper

    return decorator


# Public registry surface — re-exports placed at the bottom of the module
# to avoid spurious ruff F811 against the ``DictCache.clear`` /
# ``DictCache.info`` instance methods above (different scopes, but the
# linter conflates them).
from .content import ContentCache as ContentCache  # noqa: E402
from .disk import DiskCache as DiskCache  # noqa: E402
from .registry import clear as clear  # noqa: E402
from .registry import clear_all as clear_all  # noqa: E402
from .registry import info as info  # noqa: E402
from .registry import register as register  # noqa: E402
from .registry import unregister as unregister  # noqa: E402
