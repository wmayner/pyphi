# models/cmp.py
"""Utilities for comparing phi-objects."""

import functools
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import ClassVar
from typing import TypeVar

import numpy as np

from pyphi import utils

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

    Both ``__eq__`` and `order_by`` need to be implemented on the subclass.
    The ``order_by`` method returns a list of attributes which are compared
    to implement the ordering.

    Subclasses can optionally set a value for `unorderable_unless_eq`. This
    attribute controls whether objects are orderable: if all attributes listed
    in `unorderable_unless_eq` are not equal then the objects are not orderable
    with respect to one another and a TypeError is raised. For example: it
    doesn't make sense to compare ``Concepts`` unless they are from the same
    ``System`` or compare ``MechanismIrreducibilityAnalyses`` with different
    directions.
    """

    # The object is not orderable unless these attributes are all equal
    unorderable_unless_eq: ClassVar[list[str]] = []

    def order_by(self) -> Any:
        """Return a list of values to compare for ordering.

        The first value in the list has the greatest priority; if the first
        objects are equal the second object is compared, etc.
        """
        raise NotImplementedError

    def __lt__(self, other: object) -> bool:
        if not general_eq(self, other, self.unorderable_unless_eq):
            raise TypeError(
                f"Unorderable: the following attrs must be equal: "
                f"{self.unorderable_unless_eq}"
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

    def order_by(self) -> float:
        return self.phi


# Equality helpers
# =============================================================================


# TODO use builtin numpy methods here
def numpy_aware_eq(a: Any, b: Any) -> bool:
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    # TODO(4.0) this is broken if the iterables are sets
    if (
        (isinstance(a, Iterable) and isinstance(b, Iterable))
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):  # type: ignore[arg-type]
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b, strict=False))
    return a == b


def general_eq(a: object, b: object, attributes: Sequence[str]) -> bool:
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to |PRECISION|.
    If an attribute is called ``'mechanism'`` or ``'purview'``, it is
    compared using set equality.  All other attributes are compared with
    :func:`numpy_aware_eq`.
    """
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr in ["phi", "alpha"]:
                if not utils.eq(_a, _b):
                    return False
            elif attr in ["mechanism", "purview"]:
                if _a is None or _b is None:
                    if _a != _b:
                        return False
                elif set(_a) != set(_b):
                    return False
            elif not numpy_aware_eq(_a, _b):
                return False
        return True
    except AttributeError:
        return False
