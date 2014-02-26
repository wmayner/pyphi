#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module bundles the primary objects that power CyPhi.

"""

# TODO Optimizations:
# - Memoization
# - Preallocation
# - Vectorization
# - Cythonize the hotspots
# - Use generators instead of list comprehensions where possible for memory
#   efficiency

from .node import Node
from .network import Network
from .mechanism import Mechanism
from .subsystem import Subsystem
