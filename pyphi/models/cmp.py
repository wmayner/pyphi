#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cmp.py

"""
Utilities for comparing phi-objects.
"""

import functools
from collections import Iterable

import numpy as np

from .. import utils

# Rich comparison (ordering) helpers
# =============================================================================

def sametype(func):
    """Method decorator to return ``NotImplemented`` if the args of the wrapped
    method are of different types.

    When wrapping a rich model comparison method this will delegate (reflect)
    the comparison to the right-hand-side object, or fallback by passing it up
    the inheritance tree.
    """
    @functools.wraps(func)
    def wrapper(self, other):  # pylint: disable=missing-docstring
        if type(other) is not type(self):
            return NotImplemented
        return func(self, other)
    return wrapper


class Orderable:
    """Base mixin for implementing rich object comparisons on phi-objects.

    Both ``__eq__`` and `order_by`` need to be implemented on the subclass.
    The ``order_by`` method returns a list of attributes which are compared
    to implement the ordering.

    Subclasses can optionally set a value for `unorderable_unless_eq`. This
    attribute controls whether objects are orderable: if all attributes listed
    in `unorderable_unless_eq` are not equal then the objects are not orderable
    with respect to one another and a TypeError is raised. For example: it
    doesn't make sense to compare ``Concepts`` unless they are from the same
    ``Subsystem`` or compare ``MechanismIrreducibilityAnalyses`` with different
    directions.
    """
    # The object is not orderable unless these attributes are all equal
    unorderable_unless_eq = []

    def order_by(self):
        """Return a list of values to compare for ordering.

        The first value in the list has the greatest priority; if the first
        objects are equal the second object is compared, etc.
        """
        raise NotImplementedError

    @sametype
    def __lt__(self, other):
        if not general_eq(self, other, self.unorderable_unless_eq):
            raise TypeError(
                'Unorderable: the following attrs must be equal: {}'.format(
                    self.unorderable_unless_eq))
        return self.order_by() < other.order_by()

    @sametype
    def __le__(self, other):
        return self < other or self == other

    @sametype
    def __gt__(self, other):
        return other < self

    @sametype
    def __ge__(self, other):
        return other < self or self == other

    @sametype
    def __eq__(self, other):
        raise NotImplementedError

    @sametype
    def __ne__(self, other):
        return not self == other


# Equality helpers
# =============================================================================

# TODO use builtin numpy methods here
def numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if ((isinstance(a, Iterable) and isinstance(b, Iterable)) and
            not isinstance(a, str) and not isinstance(b, str)):
        if len(a) != len(b):
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


def general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to |PRECISION|.
    If an attribute is called ``'mechanism'`` or ``'purview'``, it is
    compared using set equality.  All other attributes are compared with
    :func:`numpy_aware_eq`.
    """
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr in ['phi', 'alpha']:
                if not utils.eq(_a, _b):
                    return False
            elif attr in ['mechanism', 'purview']:
                if _a is None or _b is None:
                    if _a != _b:
                        return False
                elif not set(_a) == set(_b):
                    return False
            else:
                if not numpy_aware_eq(_a, _b):
                    return False
        return True
    except AttributeError:
        return False
