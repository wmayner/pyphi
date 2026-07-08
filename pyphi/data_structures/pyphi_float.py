# pyright: strict
# data_structures/pyphi_float.py

from typing import Any

from pyphi.conf import config
from pyphi.utils import eq

_NUMERIC_TYPES = (int, float)


# TODO: use throughout
class PyPhiFloat(float):
    """A floating-point value compared using ``config.numerics.precision``.

    A :class:`float` subclass whose comparison operators (``==``, ``!=``, ``<``,
    ``>``, ``<=``, ``>=``) treat two values as equal when they agree to within
    the tolerance set by ``config.numerics.precision``, rather than requiring
    exact floating-point equality. This avoids spurious inequalities between φ
    values that are mathematically equal but differ slightly depending on the
    order in which floating-point arithmetic was carried out.

    Parameters
    ----------
    value : numbers.Real
        The numeric value to wrap.

    Notes
    -----
    Comparisons read ``config.numerics.precision`` at call time, through
    :func:`pyphi.utils.eq`. Hashing, in contrast, uses the precision captured
    when the instance was constructed (stored in the private ``_precision``
    attribute): ``__hash__`` returns ``hash(round(self, _precision))``, so two
    values equal to that many digits share a hash. Snapshotting the precision at
    construction keeps an instance's hash stable after it has been placed in a
    :class:`set` or used as a :class:`dict` key, even if
    ``config.numerics.precision`` is changed afterward. All other :class:`float`
    attributes and methods are inherited unchanged.

    Examples
    --------
    >>> from pyphi.data_structures.pyphi_float import PyPhiFloat
    >>> from pyphi.conf import config
    >>> config.precision = 6  # doctest: +SKIP

    Values that differ only below the precision threshold are equal:

    >>> phi1 = PyPhiFloat(0.123456789)
    >>> phi2 = PyPhiFloat(0.123456788)
    >>> phi1 == phi2  # doctest: +SKIP
    True
    >>> float(phi1) == float(phi2)  # Plain floats are not equal
    False

    The ordering operators respect the same tolerance:

    >>> PyPhiFloat(0.5) > PyPhiFloat(0.3)  # doctest: +SKIP
    True
    >>> PyPhiFloat(0.5) >= PyPhiFloat(0.5)  # doctest: +SKIP
    True

    Values within precision hash alike, so a :class:`set` deduplicates them:

    >>> phi_values = {PyPhiFloat(0.5), PyPhiFloat(0.5 + 1e-14)}
    >>> len(phi_values)
    1

    Ordering carries through :func:`min` / :func:`max`:

    >>> values = [PyPhiFloat(0.5), PyPhiFloat(0.3), PyPhiFloat(0.7)]
    >>> min(values)  # doctest: +SKIP
    PyPhiFloat(0.3)

    Serialization round-trips through :mod:`pyphi.serialize`:

    >>> from pyphi import serialize
    >>> serialize.loads(serialize.dumps(PyPhiFloat(0.5))) == PyPhiFloat(0.5)
    True
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
