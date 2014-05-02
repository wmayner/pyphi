#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys


class _Options(object):

    """User-specified options."""

    def __init__(self):
        # The number of decimal points to which phi values are considered
        # accurate
        self._PRECISION = 6
        # The threshold below which we consider differences in phi values to be
        # zero
        self.EPSILON = 10**-self._PRECISION
        # Choose whether to use multiple CPUs in evaluating subsystem cuts.
        self.PARALLEL_CUT_EVALUATION = True
        # The verbosity of parallel computation.
        self.VERBOSE_PARALLEL = 10
        # In some applications of this library, the user may prefer to define
        # single-node subsystems as having 0.5 Phi.
        self.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI = False

    # Update EPSILON whenever precision is changed
    def _set_precision(self, precision):
        self._PRECISION = precision
        self.EPSILON = 10**-precision

    def _get_precision(self):
        return self._PRECISION

    PRECISION = property(_get_precision, _set_precision,
                         "The number of decimal points to which phi values "
                         "are considered accurate.")


# Make the class look like this module
instance = _Options()
instance.__name__ = __name__
instance.__file__ = __file__
sys.modules[__name__] = instance
