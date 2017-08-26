#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

'''
Package-wide constants.
'''

import pickle
from enum import Enum

import joblib

from . import config


class Direction(Enum):
    '''Constants that parametrize cause and effect methods.

    Accessed using ``Direction.PAST`` and ``Direction.FUTURE``, etc.
    '''
    PAST = 0
    FUTURE = 1
    BIDIRECTIONAL = 2


#: The threshold below which we consider differences in phi values to be zero.
EPSILON = 10 ** - config.PRECISION

#: Label for the filesystem cache backend.
FILESYSTEM = 'fs'

#: Label for the MongoDB cache backend.
DATABASE = 'db'

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: The joblib ``Memory`` object for persistent caching without a database.
joblib_memory = joblib.Memory(cachedir=config.FS_CACHE_DIRECTORY,
                              verbose=config.FS_CACHE_VERBOSITY)

#: Node states
OFF = (0,)
ON = (1,)
