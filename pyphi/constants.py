#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Package-wide constants.
"""

import pickle
from pathlib import Path

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

DISK_CACHE_LOCATION = Path("__pyphi_cache__")

#: Node states
OFF = (0,)
ON = (1,)
