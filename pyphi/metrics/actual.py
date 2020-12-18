#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# metrics/actual.py

"""Metrics used in actual causation."""

from math import log2

from ..registry import Registry


class ActualCausationMeasureRegistry(Registry):
    """Storage for distance functions used in :mod:`pyphi.actual`.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions for use in actual causation calculations"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering an actual causation measure with PyPhi.

        Args:
            name (string): The name of the measure.

        Keyword Args:
            asymmetric (boolean): ``True`` if the measure is asymmetric.
        """

        def register_func(func):
            if asymmetric:
                self._asymmetric.append(name)
            self.store[name] = func
            return func

        return register_func

    def asymmetric(self):
        """Return a list of asymmetric measures."""
        return self._asymmetric


measures = ActualCausationMeasureRegistry()


@measures.register("PMI", asymmetric=True)
def pointwise_mutual_information(p, q):
    """Compute the pointwise mutual information (PMI).

    This is defined as

    .. math::
        \\log_2\\left(\\frac{p}{q}\\right)

    when :math:`p \\neq 0` and :math:`q \\neq 0`, and :math:`0` otherwise.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Returns:
        float: the pointwise mutual information.
    """
    if p == 0.0 or q == 0.0:
        return 0.0
    return log2(p / q)


@measures.register("WPMI", asymmetric=True)
def weighted_pointwise_mutual_information(p, q):
    """Compute the weighted pointwise mutual information (WPMI).

    This is defined as

    .. math::
        p \\log_2\\left(\\frac{p}{q}\\right)

    when :math:`p \\neq 0` and :math:`q \\neq 0`, and :math:`0` otherwise.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Returns:
        float: The weighted pointwise mutual information.
    """
    return p * pointwise_mutual_information(p, q)
