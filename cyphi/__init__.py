#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CyPhi
=====

CyPhi is a Python library for computing integrated information.
"""

__title__ = 'cyphi'
__version__ = '0.0.1'
__description__ = 'Python library for computing integrated information.',
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__author_website__ = 'http://willmayner.com'
__copyright__ = 'Copyright 2014 Will Mayner'


from .network import Network
from .subsystem import Subsystem
from . import compute
from . import utils
from . import constants


# TODO Optimizations:
# - Memoization
# - Preallocation
# - Vectorization
# - Cythonize the hotspots
# - Use generators instead of list comprehensions where possible for memory
#   efficiency
