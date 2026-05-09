# pyright: strict
# data_structures/pyphi_float.py

from typing import Any

from pyphi.conf import config
from pyphi.utils import eq

_NUMERIC_TYPES = (int, float)


# TODO(4.0) use throughout
class PyPhiFloat(float):
    """A floating-point value that's compared using config.numerics.precision.

    PyPhiFloat is a float subclass that implements precision-aware comparisons
    to avoid numerical errors when comparing phi values. All comparison operations
    (==, !=, <, >, <=, >=) use the tolerance defined by ``config.numerics.precision``
    instead of exact floating-point equality.

    This is essential for integrated information computations where values that
    are mathematically equal may differ slightly due to floating-point arithmetic.

    Args:
        value: The numeric value to wrap.

    Attributes:
        All attributes and methods of float are available.

    Note:
        The hash implementation rounds to ``config.numerics.precision`` digits to ensure
        that values equal within precision have the same hash. This makes
        PyPhiFloat safe for use in sets and as dictionary keys.

    Examples:
        Basic usage and precision-aware comparisons:

        >>> from pyphi.data_structures.pyphi_float import PyPhiFloat
        >>> from pyphi.conf import config
        >>> config.PRECISION = 6  # doctest: +SKIP

        Values that differ only at low precision are considered equal:

        >>> phi1 = PyPhiFloat(0.123456789)
        >>> phi2 = PyPhiFloat(0.123456788)
        >>> phi1 == phi2  # doctest: +SKIP
        True
        >>> float(phi1) == float(phi2)  # Plain floats are not equal
        False

        Comparison operators work as expected:

        >>> PyPhiFloat(0.5) > PyPhiFloat(0.3)  # doctest: +SKIP
        True
        >>> PyPhiFloat(0.5) >= PyPhiFloat(0.5)  # doctest: +SKIP
        True

        Hash consistency for dict/set usage (values within precision are deduplicated):

        >>> phi_values = {PyPhiFloat(0.5), PyPhiFloat(0.5 + 1e-14)}
        >>> len(phi_values)
        1

        Works transparently with min/max:

        >>> values = [PyPhiFloat(0.5), PyPhiFloat(0.3), PyPhiFloat(0.7)]
        >>> min(values)  # doctest: +SKIP
        PyPhiFloat(0.3)

        JSON serialization:

        >>> phi = PyPhiFloat(0.5)
        >>> phi.to_json()
        {'value': 0.5}
        >>> PyPhiFloat.from_json({'value': 0.5})
        0.5
    """

    # NOTE: Cannot use functools.total_ordering because it doesn't re-implement
    # existing comparison methods

    # ``_precision`` snapshots ``config.numerics.precision`` at construction time so a
    # ``PyPhiFloat`` placed in a set or dict keeps a stable hash even if
    # ``config.numerics.precision`` is later changed. The alternative — reading the
    # global at hash time — silently breaks set/dict invariants.
    _precision: int

    def __new__(cls, value: Any) -> "PyPhiFloat":
        instance = super().__new__(cls, value)
        instance._precision = int(config.numerics.precision)
        return instance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NUMERIC_TYPES):
            return NotImplemented
        return eq(self, float(other))

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _NUMERIC_TYPES):
            return NotImplemented
        return not eq(self, float(other))

    def __lt__(self, other: float) -> bool:
        return super().__lt__(other) and not eq(self, other)

    def __gt__(self, other: float) -> bool:
        return super().__gt__(other) and not eq(self, other)

    def __le__(self, other: float) -> bool:
        return super().__le__(other) or eq(self, other)

    def __ge__(self, other: float) -> bool:
        return super().__ge__(other) or eq(self, other)

    def __hash__(self) -> int:
        return hash(round(self, self._precision))

    def to_json(self) -> dict[str, float]:
        return {"value": float(self)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PyPhiFloat":
        return cls(data["value"])
