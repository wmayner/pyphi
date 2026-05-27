# models/cmp.py
"""Utilities for comparing phi-objects."""

import functools
import math
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import TypeVar

import numpy as np

# Rich comparison (ordering) helpers
# =============================================================================

T = TypeVar("T")


def sametype(func: Callable[[T, T], bool]) -> Callable[[T, object], bool | Any]:
    """Method decorator to return ``NotImplemented`` if the args of the wrapped
    method are of different types.

    When wrapping a rich model comparison method this will delegate (reflect)
    the comparison to the right-hand-side object, or fallback by passing it up
    the inheritance tree.
    """

    @functools.wraps(func)
    def wrapper(self: T, other: object) -> bool | Any:  # pylint: disable=missing-docstring
        if type(other) is not type(self):
            return NotImplemented
        return func(self, other)  # type: ignore[arg-type]

    return wrapper


class Orderable:
    """Base mixin for implementing rich object comparisons on phi-objects.

    Both ``__eq__`` and ``order_by`` need to be implemented on the subclass.
    The ``order_by`` method returns a list of attributes which are compared
    to implement the ordering.

    Subclasses can optionally override ``is_orderable_with`` to enforce
    constraints (for example, ``AcSystemIrreducibilityAnalysis`` requires
    both operands to have the same ``direction``).
    """

    def order_by(self) -> Any:
        """Return a list of values to compare for ordering.

        The first value in the list has the greatest priority; if the first
        objects are equal the second object is compared, etc.
        """
        raise NotImplementedError

    def is_orderable_with(self, other: object) -> bool:  # noqa: ARG002
        """Whether ``self`` and ``other`` are mutually orderable.

        Default: any two instances are orderable. Override in subclasses
        that need cross-instance guards.
        """
        return True

    def __lt__(self, other: object) -> bool:
        if not self.is_orderable_with(other):
            raise TypeError(
                f"Unorderable: {type(self).__name__} instances do not satisfy "
                f"the orderability constraint of this type."
            )
        return self.order_by() < other.order_by()  # type: ignore[attr-defined]

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return other < self

    def __ge__(self, other: object) -> bool:
        return other < self or self == other

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self == other


class OrderableByPhi(Orderable):
    """Mixin for implementing rich object comparisons on phi-objects that are
    ordered solely by their phi values.

    Inherits from Orderable.
    """

    phi: float  # Must be provided by subclass

    def order_by(self) -> Any:
        return self.phi


# Equality helpers
# =============================================================================


EQUALITY_TOLERANCE = 1e-13
"""Tolerance for structural equality on IIT quantities. Absorbs op-order
drift in float64 arithmetic on IIT measures while distinguishing real
math regressions. Used by `numpy_aware_eq` (model `__eq__`) and by
golden-fixture comparisons in the test suite. Independent of
`config.numerics.precision`, which governs user-configurable phi
comparison via `utils.eq`."""


def numpy_aware_eq(a: Any, b: Any) -> bool:  # noqa: PLR0911
    """Return whether two objects are equal via recursion, with float
    leaves compared up to ``EQUALITY_TOLERANCE``.

    Arrays compare via :func:`numpy.allclose`; float scalars via
    :func:`math.isclose`; other types via ``==``. Shape-mismatched or
    non-numeric arrays compare unequal rather than raising.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.allclose(a, b, rtol=EQUALITY_TOLERANCE, atol=EQUALITY_TOLERANCE)
        except (ValueError, TypeError):
            return False
    # TODO(4.0) this is broken if the iterables are sets
    if (
        (isinstance(a, Iterable) and isinstance(b, Iterable))
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):  # type: ignore[arg-type]
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        a_any: Any = a
        b_any: Any = b
        try:
            return math.isclose(
                float(a_any),
                float(b_any),
                rel_tol=EQUALITY_TOLERANCE,
                abs_tol=EQUALITY_TOLERANCE,
            )
        except (TypeError, ValueError):
            return False
    return a == b
