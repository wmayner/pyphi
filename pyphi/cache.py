#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cache.py

"""
A memory-limited cache decorator.
"""

import os
from functools import namedtuple, update_wrapper, wraps

import psutil

from . import config

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])


def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)


class _HashedSeq(list):
    """This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.
    """
    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed,
              kwd_mark=(object(),),
              fasttypes={int, str, frozenset, type(None)},
              sorted=sorted, tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments.

    The key is constructed in a way that is flat as possible rather than as a
    nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache its
    hash value, then that argument is returned without a wrapper.  This saves
    space and improves lookup speed.
    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def cache(cache={}, maxmem=config.MAXIMUM_CACHE_MEMORY_PERCENTAGE,
          typed=False):
    """Memory-limited cache decorator.

    *maxmem* is a float between 0 and 100, inclusive, specifying the maximum
    percentage of physical memory that the cache can use.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, currsize)
    with f.cache_info(). Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.
    """
    # Constants shared by all lru cache instances:
    # Unique object used to signal cache misses.
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
                    full = current_process.memory_percent() > maxmem
                misses += 1
                return result

        def cache_info():
            """Report cache statistics"""
            return _CacheInfo(hits, misses, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses, full
            cache.clear()
            hits = misses = 0
            full = False

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function


class DictCache():
    """A generic dictionary-based cache.

    Intended to be used as an object-level cache of method results.
    """
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

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
        """Get the cache key for the given function args.

        Kwargs:
           prefix: A constant to prefix to the key.
        """
        if kwargs:
            raise NotImplementedError(
                'kwarg cache keys not implemented')
        return (_prefix,) + tuple(args)


class MiceCache(DictCache):
    """A subsystem-local cache for |Mice| objects.

    Args:
        subsystem (Subsystem): The subsystem that this is a cache for
    Kwargs:
        parent_cache (MiceCache): The cache generated by the uncut
            version of ``subsystem``. Any cached |Mice| which are
            unaffected by the cut are reused in this cache. If None,
            the cache is initialized empty.
    """
    def __init__(self, subsystem, parent_cache=None):
        super(MiceCache, self).__init__()
        self.subsystem = subsystem

        if parent_cache:
            # TODO: also validate that subsystem is a
            # cut version of parent_cache.subsystem?
            # Do we need to check this at all?
            if parent_cache.subsystem.is_cut():
                raise ValueError(
                    "parent_cache must be from an uncut subsystem")
            self._build(parent_cache)

    def _build(self, parent_cache):
        """Build the initial cache from the parent.

        Only include the Mice which are unaffected by the subsystem cut.
        A Mice is affected if either the cut splits the mechanism
        or splits the connections between the purview and mechanism
        """
        for key, mice in parent_cache.cache.items():
            if not mice.damaged_by_cut(self.subsystem):
                self.cache[key] = mice

    def set(self, key, mice):
        """Set a value in the cache.

        Only cache if:
          - The subsystem is uncut (caches are only inherited from
            uncut subsystems so there is no reason to cache on cut
            subsystems.)
          - |phi| > 0. Ideally we would cache all mice, but the size
            of the cache grows way too large, making parallel computations
            incredibly inefficient because the caches have to be passed
            between process. This will be changed once global caches are
            implemented.
          - Memory is not too full.
        """
        if (not self.subsystem.is_cut() and mice.phi > 0
                and not memory_full()):
            self.cache[key] = mice

    def key(self, direction, mechanism, purviews=False, _prefix=None):
        """Cache key. This is the call signature of |find_mice|"""
        return (_prefix, direction, mechanism, purviews)


class PurviewCache(DictCache):
    """A network-level cache for possible purviews."""

    def set(self, key, value):
        """Only set if purview caching is enabled"""
        if config.CACHE_POTENTIAL_PURVIEWS:
            self.cache[key] = value


def method_cache(cache_name, key_prefix=None):
    """Caching decorator for object-level method caches.

    Cache key generation is delegated to the cache.

    Args:
        cache_name (str): The name of the (already-instantiated) cache
            on the decorated object which should be used to store results
            of this method.
        *key_prefix: A constant to use as part of the cache key in addition
            to the method arguments.
    """
    def decorator(func):
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
