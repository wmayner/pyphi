#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory
~~~~~~

Either 'fs' or 'db'.
A common interface for memoization, supporting different backends.
"""

from . import db, constants


# TODO document
class MemoizedFunc:

    """A memoized function.

    Supports two separate backends for memoization.
    """

    def __init__(self, func, ignore):
        if constants.CACHING_BACKEND == 'fs':
            # Decorate the function with the filesystem caching memoizer.
            self._memoized_func = \
                constants.joblib_memory.cache(func, ignore=ignore)
        if constants.CACHING_BACKEND == 'db':
            # Decorate the function with the database caching memoizer.
            self._memoized_func = db.memoize(func, ignore=ignore)
        # Store the raw function, without any memoization.
        self.func = func

    def __call__(self, *args, **kwargs):
        return self._memoized_func(*args, **kwargs)


def cache(ignore=[]):
    """Decorator for memoization using MemoizedFunc with a flexible backend."""
    def decorator(func):
        return MemoizedFunc(func, ignore)
    return decorator
