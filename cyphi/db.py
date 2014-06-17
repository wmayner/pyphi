#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import functools
import joblib.func_inspect
from redis import StrictRedis
from . import constants


class PickledRedis(StrictRedis):

    """A subclass of the Redis object that pickles and un-pickles objects
    before storing and retrieving."""

    def get(self, name):
        pickled_value = super(PickledRedis, self).get(name)
        if pickled_value is None:
            return None
        return pickle.loads(pickled_value)

    def set(self, name, value, ex=None, px=None, nx=False, xx=False):
        return super(PickledRedis, self).set(
            name, pickle.dumps(value, protocol=constants.PICKLE_PROTOCOL),
            ex, px, nx, xx)


# Initialize a redis instance.
instance = PickledRedis(host='localhost', port=6379, db=0)


# Bring the set and get methods of the redis instance up to the module-level
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@functools.wraps(instance.get)
def get(*args, **kwargs):
    return instance.get(*args, **kwargs)


@functools.wraps(instance.set)
def set(*args, **kwargs):
    return instance.set(*args, **kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def memoize(func, ignore=[]):
    """Decorator for memoizing a function to a redis instance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get a dictionary mapping argument names to argument values where
        # ignored arguments are omitted.
        filtered_args = joblib.func_inspect.filter_args(func, ignore, args,
                                                        kwargs)
        # Get a sorted tuple of the filtered argument.
        filtered_args = tuple(sorted(filtered_args.values()))
        # Use native hash when hashing arguments.
        key = generate_key(filtered_args)
        # Construct the key string.
        # Attempt to retrieve a precomputed value from the database.
        cached_value = instance.get(key)
        # If successful, return it.
        if cached_value is not None:
            return cached_value
        # Otherwise, compute, store, and return the value.
        else:
            result = func(*args, **kwargs)
            # Use the argument hash as the key.
            instance.set(key, result)
            return result
    return wrapper


def generate_key(value):
    # Convert the value to a (potentially singleton) tuple to be consistent
    # with joblib.filtered_args.
    return hash(tuple(value))
