# compositional_state.py

import pprint
from collections import UserDict
from copy import deepcopy
from typing import Iterable

from .direction import Direction
from .models.mechanism import Concept
from .models.subsystem import CauseEffectStructure, FlatCauseEffectStructure

DIRECTIONS = [
    Direction.CAUSE,
    Direction.EFFECT,
]


class CompositionalState(UserDict):
    """Represent assignments of purviews to mechanisms.

    Has the following structure::

        {
            <purview>: {
                Direction.CAUSE: {<mechanism>},
                Direction.EFFECT: {<mechanism>},
            }
        }

    where {<mechanism>} is a set of mechanisms. To represent a valid
    compositional state, this set must be a singleton; however, the class can be
    used with multiple mechanisms per purview & direction to keep track of
    conflicts among the distinctions in a PhiStructure.
    """

    def __init__(self, distinctions=None):
        # TODO store distinctions and use phi values in greedy conflict algo?
        # TODO deal with ties
        self.data = dict()
        if distinctions is not None:
            self.node_labels = distinctions.subsystem.node_labels
            self.update(distinctions)

    def _to_indices(self, value):
        if self.node_labels is not None:
            return self.node_labels.coerce_to_indices(value)
        return value

    def _to_labels(self, value):
        if self.node_labels is not None:
            return self.node_labels.coerce_to_labels(value)
        return value

    def __getitem__(self, key):
        return super().__getitem__(self._to_indices(key))

    def __delitem__(self, key):
        return super().__delitem__(self._to_indices(key))

    def __repr__(self):
        dict_repr = {
            self._to_labels(k): {
                kk: set(map(self._to_labels, vv)) for kk, vv in v.items()
            }
            for k, v in self.items()
        }
        return pprint.pformat(dict_repr)

    def purviews(self):
        """Return all purviews in this CompositionalState."""
        return set(self.keys())

    def mechanisms(self):
        """Return all mechanisms in this CompositionalState."""
        return set.union(
            *(set.union(*mechanism.values()) for mechanism in self.values())
        )

    def _update(self, distinction):
        if not isinstance(distinction, Concept):
            raise ValueError("distinction must be a Concept")
        for direction in DIRECTIONS:
            purview = distinction.purview(direction)
            if purview not in self.data:
                self.data[purview] = dict()
            if direction not in self.data[purview]:
                self.data[purview][direction] = set()
            self.data[purview][direction].add(distinction.mechanism)

    def update(self, value):
        """Update the CompositionalState with one or more distinctions."""
        if isinstance(value, Iterable):
            for item in value:
                self._update(item)
        else:
            self._update(value)

    def _purview_has_conflicts(self, purview):
        purview = self._to_indices(purview)
        return any(len(mechanisms) > 1 for mechanisms in self[purview].values())

    def _mechanism_has_conflicts(self, mechanism):
        mechanism = self._to_indices(mechanism)
        return any(
            any(self[purview].values()) for purview in self.conflicted_purviews()
        )

    def has_conflicts(self, purview=None, mechanism=None):
        """Return whether the CompositionalState has conflicts."""
        if purview is not None:
            return self._purview_has_conflicts(purview)
        elif mechanism is not None:
            return self._mechanism_has_conflicts(mechanism)
        else:
            return any(self.conflicted_purviews())

    def conflicted_purviews(self):
        """Return the purviews that are conflicted (specified by >1 distinction)."""
        return [purview for purview in self if self.has_conflicts(purview=purview)]

    def conflicted_mechanisms(self):
        """Return the mechanisms that are conflicted (specify >1 purview)"""
        return [
            mechanism
            for mechanism in self.mechanisms()
            if self.has_conflicts(mechanism=mechanism)
        ]

    def conflicts(self):
        """Return the subset of the CompositionalState that has conflicts."""
        conflicted = self.conflicted_purviews()
        return {
            purview: dict(mechanisms)
            for purview, mechanisms in self.items()
            if purview in conflicted
        }

    def _prune(self, empty_mechanisms=None):
        """Remove empty purview and direction slots."""
        if empty_mechanisms is None:
            empty_mechanisms = []
            for purview, both_directions in self.data.items():
                for direction, mechanisms in both_directions.items():
                    if not mechanisms:
                        empty_mechanisms.append((purview, direction))
        empty_purviews = []
        for purview, direction in empty_mechanisms:
            del self.data[purview][direction]
            if not self.data[purview]:
                empty_purviews.append(purview)
        for purview in empty_purviews:
            del self.data[purview]

    def remove(self, mechanism):
        """Remove a mechanism from this CompositionalState in-place."""
        mechanism = self._to_indices(mechanism)
        empty_mechanisms = []
        for purview, both_directions in self.items():
            for direction, mechanisms in both_directions.items():
                try:
                    mechanisms.remove(mechanism)
                except KeyError:
                    pass
                # Flag the set for removal entirely if it's now empty
                if not mechanisms:
                    empty_mechanisms.append((purview, direction))
        # Prune empty mechanism sets
        self._prune(empty_mechanisms=empty_mechanisms)

    def number_of_conflicts(self, mechanism):
        """Return the number of other mechanisms the given mechanism conflicts with."""
        mechanism = self._to_indices(mechanism)
        return sum(
            sum(
                len(mechanisms) - 1
                for mechanisms in both_directions.values()
                if mechanism in mechanisms
            )
            for both_directions in self.values()
        )

    def resolve_conflicts(self, func=None):
        """Greedily discard extra mechanisms to return a valid CompositionalState.

        Mechanisms with the most conflicts are discarded first.
        """
        resolved = deepcopy(self)
        conflicted_mechanisms = resolved.conflicted_mechanisms()
        while conflicted_mechanisms:
            # Preferentially remove mechanisms in order of their number of conflicts
            conflicted_mechanisms = sorted(
                conflicted_mechanisms, key=self.number_of_conflicts, reverse=True
            )
            most_conflicted = conflicted_mechanisms[0]
            removed = False
            if func is not None:
                # Preferentially remove mechanisms for which the given function
                # returns True
                for mechanism in conflicted_mechanisms:
                    if func(mechanism):
                        resolved.remove(mechanism)
                        removed = True
                        break
            if not removed:
                resolved.remove(most_conflicted)
            conflicted_mechanisms = resolved.conflicted_mechanisms()
        return resolved

    def conflicts_with(self, mechanism, cause_purview, effect_purview):
        """Return whether the distinction conflicts with this CompositionalState."""
        mechanism = self._to_indices(mechanism)
        purview = {
            Direction.CAUSE: self._to_indices(cause_purview),
            Direction.EFFECT: self._to_indices(effect_purview),
        }
        result = any(
            (
                purview[direction] in self
                and mechanism not in self[purview[direction]][direction]
            )
            for direction in DIRECTIONS
        )
        if not result:
            # Result will be incorrect if the CompositionalState is not
            # resolved, since the mechanisms may not be singletons
            if self.has_conflicts():
                raise ValueError(
                    "Cannot check conflict with a CompositionalState that already has conflicts!"
                )
        return result

    def conflicts_with_distinction(self, distinction):
        """Return whether the given distinction conflicts."""
        return self.conflicts_with(
            distinction.mechanism, distinction.cause.purview, distinction.effect_purview
        )

    # TODO cache?
    def purviews_of(self, mechanism):
        mechanism = self._to_indices(mechanism)
        purviews = dict()
        # Assumes each mechanism specifies only one purview in each direction
        for purview, both_directions in self.items():
            for direction, mechanisms in both_directions.items():
                if mechanism in mechanisms:
                    purviews[direction] = purview
        return purviews

    def resolve_conflicts_consistently(self, other):
        """Greedily resolve conflicts in this CompositionalState in a manner
        consistent with the other one."""

        def func(mechanism):
            purviews = self.purviews_of(mechanism)
            return other.conflicts_with(
                mechanism,
                cause_purview=purviews[Direction.CAUSE],
                effect_purview=purviews[Direction.EFFECT],
            )

        return self.resolve_conflicts(func=func)

    def filter(self, distinctions):
        """Return only the distinctions that are consistent with this CompositionalState."""
        return CauseEffectStructure(
            [
                distinction
                for distinction in distinctions
                if not self.conflicts_with_distinction(distinction)
            ],
            subsystem=distinctions.subsystem,
        )

    @classmethod
    def nonconflicting_consistent_ces_set(cls, distinction_set, reference=None):
        """Return new CESs that are nonconflicting and consistent with the first.

        Optionally specify a reference CompositionalState with which to enforce
        consistency.

        TODO: define 'consistent'
        """
        distinction_set = list(distinction_set)
        if not all(
            isinstance(distinctions, CauseEffectStructure)
            and not isinstance(distinctions, FlatCauseEffectStructure)
            for distinctions in distinction_set
        ):
            raise ValueError(
                "The given CESs must contain only CauseEffectStructures (not Flat)"
            )
        if reference is None:
            reference = cls(
                distinction_set[0],
            ).resolve_conflicts()
        return [
            cls(distinctions)
            .resolve_conflicts_consistently(reference)
            .filter(distinctions)
            for distinctions in distinction_set
        ]
