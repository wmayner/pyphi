#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    _______
#   |__   __|
#  ____| |____
# |  __   __  |    _____         ___    __     _
# | |  | |  | |   / ___/ __ __  / _ \  / /    (_)
# | |__| |__| |  / /__  / // / / ___/ / _ \  / /
# |____   ____|  \___/  \_, / /_/    /_//_/ /_/
#    __| |__           /___/
#   |_______|

"""
CyPhi
=====

CyPhi is a Python library for computing integrated information.


Options
~~~~~~~

There are several module-level options that control aspects of the computation.
They are listed here with their defaults:

- Control whether subsystem cuts should be evaluated in parallel.

    >>> import cyphi
    >>> cyphi.options.PARALLEL_CUT_EVALUATION
    True

- Verbosity level for parallel computation (0 - 100).

    >>> import cyphi
    >>> cyphi.options.VERBOSE_PARALLEL
    10

- Define the Phi value of subsystems containing only a single node with a
  self-loop to be 0.5. If set to False, their Phi will be actually be computed
  (to be zero, in this implementation).

    >>> import cyphi
    >>> cyphi.options.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI
    False
"""

__title__ = 'cyphi'
__version__ = '0.0.2'
__description__ = 'Python library for computing integrated information.',
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__author_website__ = 'http://willmayner.com'
__copyright__ = 'Copyright 2014 Will Mayner'


from .network import Network
from .subsystem import Subsystem
from . import compute, options, constants

# Create the cache if it doesn't exist
import os
os.makedirs(constants.CACHE_DIRECTORY, exist_ok=True)
