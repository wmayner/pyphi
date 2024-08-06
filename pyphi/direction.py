# direction.py
"""Causal directions."""

from enum import IntEnum, unique


@unique
class Direction(IntEnum):
    """Constant that parametrizes cause and effect methods.

    Accessed using ``Direction.CAUSE`` and ``Direction.EFFECT``, etc.
    """

    CAUSE = 0
    EFFECT = 1
    BIDIRECTIONAL = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)

    __format__ = object.__format__

    def to_json(self):
        return {"direction": self.name}

    @classmethod
    def from_json(cls, dct):
        return cls[dct["direction"]]

    def order(self, mechanism, purview):
        """Order the mechanism and purview in time.

        If the direction is ``CAUSE``, then the purview is at |t-1| and the
        mechanism is at time |t|. If the direction is ``EFFECT``, then the
        mechanism is at time |t| and the purview is at |t+1|.
        """
        if self is Direction.CAUSE:
            return purview, mechanism
        elif self is Direction.EFFECT:
            return mechanism, purview

        from . import validate

        return validate.direction(self)

    @classmethod
    def both(cls):
        return (cls.CAUSE, cls.EFFECT)

    @classmethod
    def all(cls):
        return (cls.CAUSE, cls.EFFECT, cls.BIDIRECTIONAL)

    def flip(self):
        """Return the other direction."""
        if self == Direction.CAUSE:
            return Direction.EFFECT
        else:
            return Direction.CAUSE
