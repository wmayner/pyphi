#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# constants.py
"""
Package-wide constants.
"""

import pickle
import joblib

from . import config


# The threshold below which we consider differences in phi values to be
# zero.
EPSILON = 10 ** - config.PRECISION
# Constants for accessing the past or future subspaces of concept
# space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods.
DIRECTIONS = ('past', 'future')
# Constants for labeling memoization backends.
FILESYSTEM = 'fs'
DATABASE = 'db'
# The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
# Create the joblib Memory object for persistent caching without a
# database.
joblib_memory = joblib.Memory(cachedir=config.FS_CACHE_DIRECTORY,
                              verbose=config.FS_CACHE_VERBOSITY)
