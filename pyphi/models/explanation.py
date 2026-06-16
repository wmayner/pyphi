# models/explanation.py
"""Typed explanations of why a result came out as it did (``result.explain()``).

:class:`NullResultReason` enumerates the conditions under which an analysis
yields a trivial (|big_phi| = 0 / |alpha| = 0) result. :class:`Finding` and
:class:`Explanation` are the typed account ``.explain()`` returns;
:class:`RunnerUp` is the lightweight record of the second-best partition
retained at MIP selection.
"""

from __future__ import annotations

from enum import Enum
from enum import auto
from enum import unique


@unique
class NullResultReason(Enum):
    """A condition under which an analysis returns a trivial null result."""

    # System level
    NO_SYSTEM = auto()
    NO_STRONG_CONNECTIVITY = auto()
    MONAD_WITH_NO_SELFLOOP = auto()
    MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI = auto()
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    EMPTY_CAUSE_EFFECT_STRUCTURE = auto()
    # Mechanism level
    NO_PURVIEWS = auto()
    NO_PARTITIONS = auto()
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()
    REDUCIBLE_OVER_PARTITION = auto()

    @property
    def level(self) -> str:
        """The structural level the reason arises at: ``"system"`` or
        ``"mechanism"``."""
        return _LEVEL_OF[self]


_MECHANISM_REASONS = frozenset(
    {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
    }
)

_LEVEL_OF: dict[NullResultReason, str] = {
    reason: ("mechanism" if reason in _MECHANISM_REASONS else "system")
    for reason in NullResultReason
}
