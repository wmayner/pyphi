# models/distinction.py
"""Distinction: the maximally irreducible cause and effect specified by a
mechanism (Albantakis et al. 2023). The IIT 3.0 paper terminology calls
the same object a *concept*; the alias :data:`Concept` below preserves
that vocabulary for callers using the IIT 3.0 idiom."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import numpy as np
from toolz import concat

from pyphi import utils
from pyphi import validate
from pyphi.direction import Direction

from . import cmp
from . import fmt
from .pandas import ToDictFromExplicitAttrsMixin
from .pandas import ToPandasMixin

_distinction_attributes = [
    "phi",
    "mechanism",
    "mechanism_state",
    "mechanism_label",
    "cause",
    "effect",
]


# TODO: make mechanism a property
# TODO: make phi a property
class Distinction(cmp.OrderableByPhi, ToDictFromExplicitAttrsMixin, ToPandasMixin):
    """The maximally irreducible cause and effect specified by a mechanism.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.

    Attributes:
        mechanism (tuple[int]): The mechanism that the concept consists of.
        cause (MaximallyIrreducibleCause): The |MIC| representing the
            maximally-irreducible cause of this concept.
        effect (MaximallyIrreducibleEffect): The |MIE| representing the
            maximally-irreducible effect of this concept.
        time (float): The number of seconds it took to calculate.
    """

    def __init__(
        self,
        mechanism=None,
        cause=None,
        effect=None,
    ):
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        # Attach references to this object on the cause and effect
        # TODO(4.0) document this
        assert self.cause is not None
        assert self.effect is not None
        self.cause.parent = self
        self.effect.parent = self

    def __repr__(self):
        return fmt.make_repr(self, _distinction_attributes)

    def __str__(self):
        return fmt.fmt_concept(self)

    # TODO use cached_property
    @property
    def phi(self) -> float:  # type: ignore[override]
        """float: The size of the concept.

        This is the minimum of the |small_phi| values of the concept's |MIC|
        and |MIE|.
        """
        assert self.cause is not None
        assert self.effect is not None
        return min(self.cause.phi, self.effect.phi)

    # TODO(4.0) rename?
    def mice(self, direction):
        if direction is Direction.CAUSE:
            return self.cause
        if direction is Direction.EFFECT:
            return self.effect
        validate.direction(direction)
        return None

    @property
    def cause_purview(self):
        """tuple[int]: The cause purview."""
        return getattr(self.cause, "purview", None)

    @property
    def effect_purview(self):
        """tuple[int]: The effect purview."""
        return getattr(self.effect, "purview", None)

    @cached_property
    def both_purview_unit_sets(self):
        return [
            set(self.mice(direction).purview_units)  # type: ignore[union-attr]
            for direction in Direction.both()
        ]

    @cached_property
    def purview_union(self):
        return set.union(*self.both_purview_unit_sets)

    @cached_property
    def purview_intersection(self):
        return set.intersection(*self.both_purview_unit_sets)

    @property
    def cause_repertoire(self):
        """np.ndarray: The cause repertoire."""
        return getattr(self.cause, "repertoire", None)

    @property
    def effect_repertoire(self):
        """np.ndarray: The effect repertoire."""
        return getattr(self.effect, "repertoire", None)

    @property
    def mechanism_state(self):
        """tuple(int): The state of this mechanism."""
        assert self.cause is not None
        assert self.effect is not None
        if self.cause.mechanism_state != self.effect.mechanism_state:
            raise ValueError("Inconsistent cause and effect mechanism states!")
        return self.cause.mechanism_state

    @cached_property
    def mechanism_label(self):
        """tuple[str]: The labels of the mechanism nodes."""
        return self.node_labels.label_string(self.mechanism, self.mechanism_state)  # type: ignore[arg-type]

    def purview(self, direction):
        """Return the purview in the given direction."""
        assert self.cause is not None
        assert self.effect is not None
        if direction == Direction.CAUSE:
            return self.cause.purview
        if direction == Direction.EFFECT:
            return self.effect.purview
        raise ValueError("invalid direction")

    @property
    def node_labels(self):
        assert self.cause is not None
        assert self.effect is not None
        if self.cause.node_labels != self.effect.node_labels:
            raise ValueError("Inconsistent cause and effect node labels!")
        return self.cause.node_labels

    unorderable_unless_eq: ClassVar[list[str]] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinction):
            return NotImplemented
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.mechanism_state == other.mechanism_state
            and self.cause_purview == other.cause_purview
            and self.effect_purview == other.effect_purview
            and self.eq_repertoires(other)
        )

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.mechanism,
                self.mechanism_state,
                self.cause_purview,
                self.effect_purview,
                utils.np_hash(self.cause_repertoire),
                utils.np_hash(self.effect_repertoire),
            )
        )

    def __bool__(self):
        """A concept is ``True`` if |small_phi > 0|."""
        return utils.is_positive(self.phi)

    def is_congruent(self, system_state):
        return all(
            self.mice(direction).is_congruent(system_state[direction])  # type: ignore[union-attr]
            for direction in Direction.both()
        )

    # TODO(ties) refactor
    def resolve_congruence(self, system_state):
        """Choose the MIC/MIE that are congruent, if any."""
        cause, effect = [
            next(
                filter(
                    lambda mice: mice.is_congruent(system_state[direction]),  # type: ignore[union-attr]
                    concat(
                        filter(
                            None,
                            [
                                self.mice(direction).state_ties,  # type: ignore[union-attr]
                                self.mice(direction).purview_ties,  # type: ignore[union-attr]
                            ],
                        )
                    ),
                ),
                None,
            )
            for direction in Direction.both()
        ]
        if cause is None or effect is None:
            return None
        return type(self)(
            mechanism=self.mechanism,
            cause=cause,
            effect=effect,
        )

    def eq_repertoires(self, other):
        """Return whether this concept has the same repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that the mechanism
            and purview indices refer to, might be different.
        """
        return np.array_equal(
            self.cause_repertoire,  # pyright: ignore[reportArgumentType]
            other.cause_repertoire,  # type: ignore[arg-type]
        ) and np.array_equal(
            self.effect_repertoire,  # pyright: ignore[reportArgumentType]
            other.effect_repertoire,  # type: ignore[arg-type]
        )

    def emd_eq(self, other):
        """Return whether this concept is equal to another in the context of
        an EMD calculation.
        """
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.eq_repertoires(other)
        )

    _dict_attrs = _distinction_attributes

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "mechanism": self.mechanism,
            "cause": self.cause,
            "effect": self.effect,
        }

    @classmethod
    def from_json(cls, dct):
        instance = cls(**dct)
        # Restore parent references to MICEs
        assert instance.cause is not None
        assert instance.effect is not None
        instance.cause.parent = instance
        instance.effect.parent = instance
        return instance

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore parent references to MICEs
        assert self.cause is not None
        assert self.effect is not None
        self.cause.parent = self
        self.effect.parent = self


# IIT 3.0 paper terminology calls a distinction a "concept". The alias
# preserves that vocabulary for callers using the IIT 3.0 idiom; the
# runtime class is identical.
Concept = Distinction
