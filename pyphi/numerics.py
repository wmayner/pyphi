# numerics.py
"""Tolerant scalar comparison of φ, Φ, and α values.

Floating-point results that are mathematically equal can differ by
roughly 1e-15 when computed through different code paths: distinct
algebraic routes to the same value produce different bit patterns, and
summation order is not associative. Integrated-information theory
treats ties between candidates (partitions, purviews, states, systems)
as meaningful, so detecting them requires comparison up to a tolerance
rather than exact equality.

These predicates are the only tolerant scalar comparisons in the
library. Values themselves are plain :class:`float`\\ s with exact
comparison semantics; tolerance applies where a comparison decides an
outcome. Selection among competing φ-objects goes through
:mod:`pyphi.resolve_ties`, which clusters candidates with :func:`eq`.

The tolerance is ``10**-precision`` with ``precision`` read from
``config.numerics.precision`` at call time (default 13, roughly two
orders of magnitude above the observed noise floor and far below
genuine φ differences).
"""

import math

import numpy as np

from .conf import config


def _epsilon() -> float:
    return 10 ** (-int(config.numerics.precision))


def eq(x: float, y: float) -> bool:
    """Return whether two values are equal up to ``config.numerics.precision``."""
    epsilon = _epsilon()
    return math.isclose(x, y, rel_tol=epsilon, abs_tol=epsilon)


def lt(x: float, y: float) -> bool:
    """Return whether ``x`` is less than ``y`` beyond
    ``config.numerics.precision``: strictly less and not within the
    tolerance of equal. For the reversed comparisons use ``lt(y, x)`` /
    ``le(y, x)``."""
    # Need ``bool`` to cast from numpy to native Boolean
    return not eq(x, y) and bool(x < y)


def le(x: float, y: float) -> bool:
    """Return whether ``x`` is less than ``y`` or equal to it up to
    ``config.numerics.precision``."""
    # Need ``bool`` to cast from numpy to native Boolean
    return eq(x, y) or bool(x < y)


def is_zero(x: float) -> bool:
    """Return whether ``x`` is zero up to ``config.numerics.precision``."""
    return eq(x, 0.0)


def is_positive(x: float) -> bool:
    """Return whether ``x`` is positive up to ``config.numerics.precision``."""
    # Need ``bool`` to cast from numpy to native Boolean
    return not eq(x, 0) and bool(x > 0)


def is_nonpositive(x: float) -> bool:
    """Return whether ``x`` is nonpositive (exact)."""
    # Need ``bool`` to cast from numpy to native Boolean
    return bool(x <= 0)


def eq_mask(array: np.ndarray, value: float) -> np.ndarray:
    """Return a boolean mask of the elements equal to ``value`` up to
    ``config.numerics.precision``.

    Elementwise-equivalent to :func:`eq`. The comparison replicates
    ``math.isclose`` — a symmetric relative tolerance with an absolute
    floor, and non-finite values equal only to themselves — which differs
    from ``np.isclose``'s asymmetric additive form.
    """
    epsilon = _epsilon()
    a = np.asarray(array, dtype=float)
    if not math.isfinite(value):
        return a == value
    tol = np.maximum(epsilon * np.maximum(np.abs(a), abs(value)), epsilon)
    return np.isfinite(a) & (np.abs(a - value) <= tol)


def positive_mask(array: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the elements positive up to
    ``config.numerics.precision``.

    Equivalent to applying :func:`is_positive` elementwise. Values within
    the tolerance of zero (for example, the surprisal ``-log2(p)`` of a
    probability that is 1 up to floating-point noise) are masked out.
    """
    epsilon = _epsilon()
    a = np.asarray(array)
    return (a > 0) & ~np.isclose(a, 0.0, rtol=epsilon, atol=epsilon)


def round_to_precision(x: float) -> float:
    """Return ``x`` rounded to ``config.numerics.precision`` decimal places."""
    return round(x, int(config.numerics.precision))
