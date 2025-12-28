# pyright: strict
# direction.py
"""Causal directions."""

from enum import IntEnum
from enum import unique


@unique
class Direction(IntEnum):
    """Constant that parametrizes cause and effect methods.

    Accessed using ``Direction.CAUSE`` and ``Direction.EFFECT``, etc.
    """

    CAUSE = 0
    EFFECT = 1
    BIDIRECTIONAL = 2

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return repr(self)

    __format__ = object.__format__

    def to_json(self) -> dict[str, str]:
        return {"direction": self.name}

    @classmethod
    def from_json(cls, dct: dict[str, str]) -> "Direction":
        return cls[dct["direction"]]

    def order(
        self, mechanism: tuple[int, ...], purview: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Order the mechanism and purview in time.

        If the direction is ``CAUSE``, then the purview is at |t-1| and the
        mechanism is at time |t|. If the direction is ``EFFECT``, then the
        mechanism is at time |t| and the purview is at |t+1|.
        """
        if self is Direction.CAUSE:
            return purview, mechanism
        if self is Direction.EFFECT:
            return mechanism, purview

        from . import validate

        validate.direction(self)
        # This should never be reached; validate.direction raises for invalid directions
        raise AssertionError(f"Unexpected direction: {self}")

    @classmethod
    def both(cls) -> tuple["Direction", "Direction"]:
        return (cls.CAUSE, cls.EFFECT)

    @classmethod
    def all(cls) -> tuple["Direction", "Direction", "Direction"]:
        return (cls.CAUSE, cls.EFFECT, cls.BIDIRECTIONAL)

    def flip(self) -> "Direction":
        """Return the other direction."""
        if self == Direction.CAUSE:
            return Direction.EFFECT
        return Direction.CAUSE
