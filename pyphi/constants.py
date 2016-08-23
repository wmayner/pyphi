#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Package-wide constants.
"""

import pickle

import joblib

from . import config

#: The threshold below which we consider differences in phi values to be zero.
EPSILON = 10 ** - config.PRECISION

#: Key for accessing the cause side of computations and concept space.
PAST = 0

#: Key for accessing the effect side of computations and concept space.
FUTURE = 1

#: Constants which parametrize cause and effect methods.
#: Accessed using ``DIRECTIONS[PAST]`` and ```DIRECTIONS[FUTURE]``
DIRECTIONS = ('past', 'future')

#: Label for the filesystem cache backend.
FILESYSTEM = 'fs'

#: Label for the MongoDB cache backed.
DATABASE = 'db'

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: The joblib Memory object for persistent caching without a database.
joblib_memory = joblib.Memory(cachedir=config.FS_CACHE_DIRECTORY,
                              verbose=config.FS_CACHE_VERBOSITY)
