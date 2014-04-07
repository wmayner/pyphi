#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CyPhi
=====

CyPhi is a Cython library for computing integrated information.
"""

__title__ = 'cyphi'
__version__ = '0.0.1'
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__copyright__ = 'Copyright 2014 Will Mayner'


import numpy as np

from .network import Network
from .subsystem import Subsystem
from . import utils
from . import constants

# TODO Optimizations:
# - Memoization
# - Preallocation
# - Vectorization
# - Cythonize the hotspots
# - Use generators instead of list comprehensions where possible for memory
#   efficiency
