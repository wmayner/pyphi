#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib

# Constants for accessing the past or future subspaces of concept
# space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods.
DIRECTIONS = ('past', 'future')

# Caching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The maximum percentage of RAM that CyPhi should use for caching.
# Defaults to 50%.
MAXMEM = 50
# Constants for labeling memoization backends.
FILESYSTEM = 'fs'
DATABASE = 'db'
# Directory for the persistent joblib Memory cache.
CACHE_DIRECTORY = '__cyphi_cache__'
# The protocol used for pickling objects.
PICKLE_PROTOCOL = 4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The joblib Memory object for persistent caching
joblib_memory = joblib.Memory(cachedir=CACHE_DIRECTORY, verbose=1)
