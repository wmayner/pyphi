#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import errno
import json
import joblib
from collections import namedtuple

# Constants for accessing the past or future subspaces of concept
# space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods.
DIRECTIONS = ('past', 'future')

# Caching / Database
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The maximum percentage of RAM that CyPhi should use for caching.
# Defaults to 50%.
MAXMEM = 50
# Constants for labeling memoization backends.
FILESYSTEM = 'fs'
DATABASE = 'db'
# Directory for the persistent joblib Memory cache.
JOBLIB_CACHE_DIRECTORY = '__cyphi_cache__'
# The protocol used for pickling objects.
PICKLE_PROTOCOL = 4
# TODO: document redis config
# Parse and set the redis configuration, defaulting to ``redis-server``'s
# default settings if no configuration file is in the current directory.
mongo_config = namedtuple('MongoConfig',
                          ['HOST', 'PORT', 'DATABASE_NAME', 'COLLECTION_NAME'])
MONGO_CONFIG = mongo_config(HOST='localhost', PORT=27017,
                            DATABASE_NAME='cyphi', COLLECTION_NAME='cache')
MONGO_CONFIG_FILE = 'mongo_config.json'
try:
    with open(MONGO_CONFIG_FILE) as f:
        data = json.load(f)
        print("[CyPhi] Loaded redis configuration from file:\n\t", data)
except OSError as e:
    if e.errno == errno.ENOENT:
        data = {'host': 'localhost', 'port': '6379', 'db': 0}
        print("[CyPhi] Using default MongoDB configuration (no config file "
              "provided):\n\t", MONGO_CONFIG)
    else:
        raise e
# TODO: Use proper logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The joblib Memory object for persistent caching
joblib_memory = joblib.Memory(cachedir=JOBLIB_CACHE_DIRECTORY, verbose=1)
