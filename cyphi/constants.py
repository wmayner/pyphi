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

# Maximum number of entries in an @lru_cache-decorated function.
# This is to avoid thrashing.
#
#   NOTE: Magic number 1.5 million chosen according to these
#   assumptions:
#   TODO! find a non-magic-number solution
#
#   - a conservative estimate of 512 bytes as the average size of numpy
#     arrays returned by `cause`_repertoire and `effect_repertoire`
#     (where the vast majority of distinct calls will be);
#   - a system with 14GB of RAM;
#   - the number of calls to `cause_repertoire` and `effect_repertoire`
#     are equal.
#
#   So, in order to limit memory usage to below 14G, we have
#       14GB / 512b per call ~= 30,000,000 calls, and
#       30,000,000 calls / 2 repertoire functions
#           = 15,000,000 calls per repertoire function
LRU_CACHE_MAX_SIZE = 15000000
