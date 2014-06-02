#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from joblib import Memory

# Constants for accessing the past or future subspaces of concept
# space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods
DIRECTIONS = ('past', 'future')
# Directory for the persistent joblib Memory cache
CACHE_DIRECTORY = '__cyphi_cache__'
# The maximum percentage of RAM that CyPhi should use for caching.
# Defaults to 50%.
MAXMEM = 50


# The joblib Memory object for persistent caching
memory = Memory(cachedir=CACHE_DIRECTORY, verbose=0)
