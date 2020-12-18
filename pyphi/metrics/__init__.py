#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# metrics/__init__.py

"""
Functions for measuring distances.
"""

from ..registry import Registry


class MeasureRegistry(Registry):
    """Storage for measures registered with PyPhi.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.

    For actual causation calculations, use
    ``config.ACTUAL_CAUSATION_MEASURE``.
    """

    # pylint: disable=arguments-differ

    desc = "measures"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering a measure with PyPhi.

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


measures = MeasureRegistry()
