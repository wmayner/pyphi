#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory
~~~~~~

Either 'fs' or 'db'.
A common interface for memoization, supporting different backends.
"""
# TODO document

from . import db, constants


# Default to using the local filesystem as the caching backend.
DEFAULT = constants.DATABASE
# The backend currently being used.
BACKEND = DEFAULT


# A list of all instances of MemoizedFuncs, so this singleton can set
# their backends.
_memoized_funcs = []


def set_backend(backend):
    """Change the backend that CyPhi uses for memoization."""
    global BACKEND
    BACKEND = backend
    for f in _memoized_funcs:
        f.set_backend(BACKEND)


def _register(memoized_func):
    _memoized_funcs.append(memoized_func)


class MemoizedFunc:

    """A memoized function.

    Supports two separate backends for memoization.
    """

    def __init__(self, func, ignore):
        # Register this instance with the memory singleton.
        _register(self)
        # The backend this function is currently using.
        self.backend = DEFAULT
        # Memoize the function with the database.
        fs_memoized = constants.joblib_memory.cache(func, ignore=ignore)
        # Memoize the function with joblib's Memory caching.
        db_memoized = db.memoize(func, ignore=ignore)
        # Store the memoized functions as attributes.
        self.memoizations = {}
        self.memoizations[constants.DATABASE] = db_memoized
        self.memoizations[constants.FILESYSTEM] = fs_memoized
        # This will be the function that's actually called; it can refer to
        # any of the memoizations.
        self._memoized_func = self.memoizations[DEFAULT]
        # Store the raw function, without any memoization.
        self.func = func

    def set_backend(self, backend):
        self.backend = backend
        # Change which memoization is called, according to the new backend.
        self._memoized_func = self.memoizations[backend]

    def __call__(self, *args, **kwargs):
        return self._memoized_func(*args, **kwargs)


def cache(ignore=[]):
    """Decorator for memoization using MemoizedFunc with a flexible backend."""
    def decorator(func):
        return MemoizedFunc(func, ignore)
    return decorator
