#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pymongo
from bson.binary import Binary
import functools
from collections import Iterable
import joblib.func_inspect
from . import constants

client = pymongo.MongoClient(constants.MONGODB_CONFIG['host'],
                             constants.MONGODB_CONFIG['port'])
database = client[constants.MONGODB_CONFIG['database_name']]
collection = database[constants.MONGODB_CONFIG['collection_name']]
KEY_FIELD = 'k'
VALUE_FIELD = 'v'
# Index documents by their keys. Enforce that the keys be unique.
collection.create_index('k', unique=True)


def get(key):
    docs = list(collection.find({KEY_FIELD: key}))
    # Return None if we didn't find anything.
    if not docs:
        return None
    pickled_value = docs[0][VALUE_FIELD]
    # Unpickle and return the value.
    return pickle.loads(pickled_value)


def set(key, value):
    # Pickle the value.
    value = pickle.dumps(value, protocol=constants.PICKLE_PROTOCOL)
    # Store the value as binary data in a document.
    doc = {
        KEY_FIELD: key,
        VALUE_FIELD: Binary(value)
    }
    # Pickle and store the value with its key. If the key already exists, we
    # don't insert (since the key is a unique index), and we don't care.
    try:
        return collection.insert(doc)
    except pymongo.errors.DuplicateKeyError:
        return None


def memoize(func, ignore=[]):
    """Decorator for memoizing a function to a database."""
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
        cached_value = get(key)
        # If successful, return it.
        if cached_value is not None:
            return cached_value
        # Otherwise, compute, store, and return the value.
        else:
            result = func(*args, **kwargs)
            # Use the argument hash as the key.
            set(key, result)
            return result
    return wrapper


# TODO!!!: check this singleton tuple business
def generate_key(value):
    # Convert the value to a (potentially singleton) tuple to be consistent
    # with joblib.filtered_args.
    if isinstance(value, Iterable):
        return hash(tuple(value))
    else:
        return hash((value, ))
