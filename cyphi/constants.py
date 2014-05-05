#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Constants for accessing the past or future subspaces of concept
# space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods
DIRECTIONS = ('past', 'future')
# Directory for the persistent joblib Memory cache
CACHE_DIRECTORY = '__cyphi_cache__'
# The amount of memory in bytes that CyPhi should leave available on
# the system. Defaults to 2G.
GB = 1024 ** 3
USE_MEMORY_UP_TO = 2 * GB
