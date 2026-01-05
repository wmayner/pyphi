# cache/cache_utils.py
"""Common utilities for caching."""

import os
from collections import namedtuple

import psutil

from pyphi.conf import config

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])


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

    def __hash__(self):  # type: ignore[override]
        return self.hashvalue


def _make_key(
    args,
    kwds,
    typed,
    kwd_mark=(object(),),
    fasttypes=None,
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
    if fasttypes is None:
        fasttypes = {int, str, frozenset, type(None)}
    key = args
    sorted_items = None
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            assert sorted_items is not None  # Type narrowing: kwds is truthy
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)
