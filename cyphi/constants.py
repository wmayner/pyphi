#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# TODO move precision to options


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
        # Directory for the persistent joblib Memory cache
        self.CACHE_DIRECTORY = '__cyphi_cache__'

        # Maximum number of entries in an @lru_cache-decorated function.
        # This is to avoid thrashing.
        #
        #   NOTE: Magic number 1.5 million chosen according to these
        #   assumptions:
        #   TODO! find a non-magic-number solution
        #
        #   - a conservative estimate of 512 bytes as the average size of numpy
        #     arrays returned by `cause`_repertoire and `effect_repertoire`
        #     (where the vast majority of distinct calls will be);
        #   - a system with 14GB of RAM;
        #   - the number of calls to `cause_repertoire` and `effect_repertoire`
        #     are equal.
        #
        #   So, in order to limit memory usage to below 14G, we have
        #       14GB / 512b per call ~= 30,000,000 calls, and
        #       30,000,000 calls / 2 repertoire functions
        #           = 15,000,000 calls per repertoire function
        self.LRU_CACHE_MAX_SIZE = 15000000

    # Update EPSILON whenever precision is changed
    def set_precision(self, precision):
        self._PRECISION = precision
        self.EPSILON = 10**-precision

    def get_precision(self):
        return self._PRECISION

    PRECISION = property(get_precision, set_precision,
                         "The number of decimal points to which phi values "
                         "are considered accurate.")


# Make the class look like this module
instance = _Constants()
instance.__name__ = __name__
instance.__file__ = __file__
sys.modules[__name__] = instance
