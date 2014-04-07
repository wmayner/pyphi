#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil

# The number of cores available for parallel computation
NUMBER_OF_CORES = psutil.NUM_CPUS
# The number of decimal points to which phi values are considered accurate
PRECISION = 6
# The threshold below which we consider differences in phi values to be zero
EPSILON = 10**-PRECISION
# Constants for accessing the past or future subspaces of concept space.
PAST = 0
FUTURE = 1
# Constants for using cause and effect methods
DIRECTIONS = ('past', 'future')
