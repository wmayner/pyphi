#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cache.py

"""
Memoization and caching utilities.
"""

# pylint: disable=redefined-builtin,redefined-outer-name,missing-docstring
# pylint: disable=no-self-use,arguments-differ
# pylint: disable=dangerous-default-value,redefined-builtin
# pylint: disable=abstract-method

import os
import pickle
from functools import namedtuple, update_wrapper, wraps

import joblib
import psutil
import redis

from . import constants
from .conf import config

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])

joblib_memory = joblib.Memory(location=constants.DISK_CACHE_LOCATION, verbose=0)


def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return current_process.memory_percent() > config.MAXIMUM_CACHE_MEMORY_PERCENTAGE


class _HashedSeq(list):
    """This class guarantees that ``hash()`` will be called no more than once
    per element.  This is important because the ``lru_cache()`` will hash the
    key multiple times on a cache miss.
    """

    __slots__ = ("hashvalue",)

    def __init__(self, tup, hash=hash):
        super().__init__()
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(
    args,
    kwds,
    typed,
    kwd_mark=(object(),),
    fasttypes={int, str, frozenset, type(None)},
    sorted=sorted,
    tuple=tuple,
    type=type,
    len=len,
):
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


def cache(cache={}, maxmem=config.MAXIMUM_CACHE_MEMORY_PERCENTAGE, typed=False):
    """Memory-limited cache decorator.

    ``maxmem`` is a float between 0 and 100, inclusive, specifying the maximum
    percentage of physical memory that the cache can use.

    If ``typed`` is ``True``, arguments of different types will be cached
    separately. For example, f(3.0) and f(3) will be treated as distinct calls
    with distinct results.

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
            """Report cache statistics."""
            return _CacheInfo(hits, misses, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics."""
            nonlocal hits, misses, full
            cache.clear()
            hits = misses = 0
            full = False

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function


class DictCache:
    """A generic dictionary-based cache.

    Intended to be used as an object-level cache of method results.
    """

    def __init__(self, cache=None, hits=0, misses=0):
        self.cache = dict() if cache is None else cache
        self.hits = hits
        self.misses = misses

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
            raise NotImplementedError("kwarg cache keys not implemented")
        return (_prefix,) + tuple(args)
    
    def __repr__(self):
        return "{}(cache={}, hits={}, misses={})".format(
            type(self).__name__, self.cache, self.hits, self.misses
        )


def redis_init(db):
    return redis.StrictRedis(
        host=config.REDIS_CONFIG["host"], port=config.REDIS_CONFIG["port"], db=db
    )


# Expose the StrictRedis API, maintaining one connection pool
# The connection pool is multi-process safe, and is reinitialized when the
# client detects a fork. See:
# https://github.com/andymccurdy/redis-py/blob/5109cb4f/redis/connection.py#L950
#
# TODO: rebuild connection after config changes?
redis_conn = redis_init(config.REDIS_CONFIG["db"])


def redis_available():
    """Check if the Redis server is connected."""
    try:
        return redis_conn.ping()
    except redis.exceptions.ConnectionError:
        return False


# TODO: use a cache prefix?
# TODO: key schema for easy access/queries
class RedisCache:
    def clear(self):
        """Flush the cache."""
        redis_conn.flushdb()
        redis_conn.config_resetstat()

    @staticmethod
    def size():
        """Size of the Redis cache.

        .. note:: This is the size of the entire Redis database.
        """
        return redis_conn.dbsize()

    def info(self):
        """Return cache information.

        .. note:: This is not the cache info for the entire Redis key space.
        """
        info = redis_conn.info()
        return _CacheInfo(info["keyspace_hits"], info["keyspace_misses"], self.size())

    def get(self, key):
        """Get a value from the cache.

        Returns None if the key is not in the cache.
        """
        value = redis_conn.get(key)

        if value is not None:
            value = pickle.loads(value)

        return value

    def set(self, key, value):
        """Set a value in the cache."""
        value = pickle.dumps(value, protocol=constants.PICKLE_PROTOCOL)
        redis_conn.set(key, value)

    def key(self):
        """Delegate to subclasses."""
        raise NotImplementedError


def validate_parent_cache(parent_cache):
    # TODO: also validate that subsystem is a cut version of
    # parent_cache.subsystem? Do we need to check this at all?
    if parent_cache.subsystem.is_cut:
        raise ValueError("parent_cache must be from an uncut subsystem")


class PurviewCache(DictCache):
    """A network-level cache for possible purviews."""

    def set(self, key, value):
        """Only set if purview caching is enabled"""
        if config.CACHE_POTENTIAL_PURVIEWS:
            self.cache[key] = value


def method(cache_name, key_prefix=None):
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
        if not config.CACHE_REPERTOIRES and "repertoire" in func.__name__:
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
