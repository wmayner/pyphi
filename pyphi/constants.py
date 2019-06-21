#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Package-wide constants.
"""

import pickle

#: The threshold below which we consider differences in phi values to be zero.
EPSILON = None
# NOTE: This is set dynamically by `conf.py` when PRECISION is changed; see
# `conf.py` for default value.

#: Label for the filesystem cache backend.
FILESYSTEM = 'fs'

#: Label for the MongoDB cache backend.
DATABASE = 'db'

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: The joblib ``Memory`` object for persistent caching without a database.
joblib_memory = None
# NOTE: This is set dynamically by `conf.py` when PRECISION is changed; see
# `conf.py` for default value.

#: Node states
OFF = (0,)
ON = (1,)
