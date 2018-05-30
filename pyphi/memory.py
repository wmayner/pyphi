#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# memory.py

"""
Decorators and objects for memoization.
"""

# pylint: disable=missing-docstring

import functools

import joblib.func_inspect

from . import config, constants, db


def cache(ignore=None):
    """Decorator for memoizing a function using either the filesystem or a
    database.
    """
    def decorator(func):
        # Initialize both cached versions
        joblib_cached = constants.joblib_memory.cache(func, ignore=ignore)
        db_cached = DbMemoizedFunc(func, ignore)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Dynamically choose the cache at call-time, not at import."""
            if func.__name__ == '_sia' and not config.CACHE_SIAS:
                f = func
            elif config.CACHING_BACKEND == 'fs':
                f = joblib_cached
            elif config.CACHING_BACKEND == 'db':
                f = db_cached
            return f(*args, **kwargs)

        return wrapper
    return decorator


class DbMemoizedFunc:
    """A memoized function, with a databse backing the cache."""

    def __init__(self, func, ignore):
        # Store a reference to the raw function, without any memoization.
        self.func = func
        # The list of arguments to ignore when getting cache keys.
        self.ignore = ignore if ignore is not None else []

        # This is the memoized function.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = self.get_output_key(args, kwargs)
            # Attempt to retrieve a precomputed value from the database.
            cached_value = db.find(key)
            # If successful, return it.
            if cached_value is not None:
                return cached_value
            # Otherwise, compute, store, and return the value.
            result = func(*args, **kwargs)
            # Use the argument hash as the key.
            db.insert(key, result)
            return result

        # Store the memoized function.
        self._memoized_func = wrapper

    def __call__(self, *args, **kwargs):
        return self._memoized_func(*args, **kwargs)

    # TODO make this easier to use
    def get_output_key(self, args, kwargs):
        """Return the key that the output should be cached with, given
        arguments, keyword arguments, and a list of arguments to ignore.
        """
        # Get a dictionary mapping argument names to argument values where
        # ignored arguments are omitted.
        filtered_args = joblib.func_inspect.filter_args(
            self.func, self.ignore, args, kwargs)
        # Get a sorted tuple of the filtered argument.
        filtered_args = tuple(sorted(filtered_args.values()))
        # Use native hash when hashing arguments.
        return db.generate_key(filtered_args)

    def load_output(self, args, kwargs):
        """Return cached output."""
        return db.find(self.get_output_key(args, kwargs))
