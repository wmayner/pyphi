#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys


class _Constants(object):

    def __init__(self):
        # The number of decimal points to which phi values are considered
        # accurate
        self._PRECISION = 6
        # The threshold below which we consider differences in phi values to be
        # zero
        self.EPSILON = 10**-self._PRECISION
        # Constants for accessing the past or future subspaces of concept
        # space.
        self.PAST = 0
        self.FUTURE = 1
        # Constants for using cause and effect methods
        self.DIRECTIONS = ('past', 'future')

    # Update EPSILON whenever precision is changed
    def set_precision(self, precision):
        self._PRECISION = precision
        self.EPSILON = 10**-precision

    def get_precision(self):
        return self._PRECISION

    PRECISION = property(get_precision, set_precision,
                         "The number of decimal points to which phi values " +
                         " are considered accurate")


# Make the class look like this module
instance = _Constants()
instance.__name__ = __name__
instance.__file__ = __file__
sys.modules[__name__] = instance
