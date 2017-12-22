#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# db.py

"""
Interface to MongoDB that exposes it as a key-value store.
"""

import pickle
from collections import Iterable

import pymongo
from bson.binary import Binary

from . import config, constants

KEY_FIELD = 'k'
VALUE_FIELD = 'v'


# Initialize dummy database API objects.
client, database, collection = None, None, None
# Connect to MongoDB if the caching backend is set to 'db'.
if config.CACHING_BACKEND == 'db':
    # TODO: use reconnect proxy
    client = pymongo.MongoClient(config.MONGODB_CONFIG['host'],
                                 config.MONGODB_CONFIG['port'])
    database = client[config.MONGODB_CONFIG['database_name']]
    collection = database[config.MONGODB_CONFIG['collection_name']]
    # Index documents by their keys. Enforce that the keys be unique.
    collection.create_index('k', unique=True)


def find(key):
    """Return the value associated with a key.

    If there is no value with the given key, returns ``None``.
    """
    docs = list(collection.find({KEY_FIELD: key}))
    # Return None if we didn't find anything.
    if not docs:
        return None
    pickled_value = docs[0][VALUE_FIELD]
    # Unpickle and return the value.
    return pickle.loads(pickled_value)


def insert(key, value):
    """Store a value with a key.

    If the key is already present in the database, this does nothing.
    """
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


# TODO: check this singleton tuple business
def generate_key(filtered_args):
    """Get a key from some input.

    This function should be used whenever a key is needed, to keep keys
    consistent.
    """
    # Convert the value to a (potentially singleton) tuple to be consistent
    # with joblib.filtered_args.
    if isinstance(filtered_args, Iterable):
        return hash(tuple(filtered_args))
    return hash((filtered_args,))
