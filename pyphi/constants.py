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

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

DISK_CACHE_LOCATION = "__pyphi_cache__"

#: Node states
OFF = (0,)
ON = (1,)
