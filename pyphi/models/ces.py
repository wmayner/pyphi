# models/ces.py
"""``CauseEffectStructure`` — a collection of distinctions/concepts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence

from toolz import concat

from pyphi import utils
from pyphi.conf import fallback
from pyphi.direction import Direction

from . import cmp
from . import fmt
from .pandas import ToPandasMixin
from .state_specification import SystemStateSpecification


def _concept_sort_key(concept):
    return (len(concept.mechanism), concept.mechanism)


def defaultdict_set():
    return defaultdict(set)


def _purview_inclusion(distinction_attr, distinctions, min_order, max_order):
    purview_inclusion_by_order = defaultdict(defaultdict_set)
    for distinction in distinctions:
        for subset in map(
            frozenset,
            utils.powerset(
                getattr(distinction, distinction_attr),
                nonempty=True,
                min_size=min_order,
                max_size=max_order,
            ),
        ):
            purview_inclusion_by_order[len(subset)][subset].add(distinction)
    return purview_inclusion_by_order


def _find_multiplicities(func, distinctions):
    """Return a mapping from purviews to multiplicities of the values of ``func``."""
    multiplicities = defaultdict_set()
    for d in distinctions:
        for direction in Direction.both():
            multiplicities[d.purview(direction)].add(func(d.mice(direction)))
    return multiplicities


def _get_mechanism(mice):
    return mice.mechanism


def _get_state(mice):
    return mice.specified_state.state


class CauseEffectStructure(cmp.Orderable, Sequence, ToPandasMixin):
    """A collection of concepts."""

    def __init__(self, concepts=(), resolved_congruence=False):
        # Normalize the order of concepts
        # TODO(4.0) convert to set?
        self.concepts = tuple(sorted(concepts, key=_concept_sort_key))
        self._specifiers = None
        self._purview_inclusion_by_order = defaultdict(defaultdict_set)
        # Flag to indicate whether distinctions have been filtered according to
        # congruence with a SIA specified state
        # TODO(4.0) use a subclass instead, as with MICE?
        self._resolved_congruence = resolved_congruence
        self._sum_phi = None

    def __len__(self):
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts)

    def __getitem__(self, value):
        if isinstance(value, slice):
            return type(self)(self.concepts[value])
        return self.concepts[value]

    def __repr__(self):
        return fmt.make_repr(self, ["concepts"])

    def __str__(self):
        return fmt.fmt_ces(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseEffectStructure):
            return NotImplemented
        return self.concepts == other.concepts

    def __hash__(self):
        return hash(self.concepts)

    def order_by(self):
        return [self.concepts]

    def to_json(self):
        return {"concepts": self.concepts}

    @property
    def flat(self):
        """An iterator over causes and effects (one ``MICE`` per direction
        per concept), for callers that want to operate at the MICE level
        rather than the concept level.
        """
        return concat([concept.cause, concept.effect] for concept in self)

    def sum_phi(self):
        if self._sum_phi is None:
            self._sum_phi = sum(self.phis)
        return self._sum_phi

    @property
    def phis(self):
        """The |small_phi| values of each concept."""
        for concept in self:
            yield concept.phi

    @property
    def mechanisms(self):
        """The mechanism of each concept."""
        for concept in self:
            yield concept.mechanism

    def _purviews(self, direction):
        for concept in self:
            yield concept.purview(direction)

    def purviews(self, direction):
        """Return the purview of each concept in the given direction."""
        if isinstance(direction, Iterable):
            for _direction in direction:
                yield from self._purviews(_direction)
        else:
            yield from self._purviews(direction)

    @property
    def labeled_mechanisms(self):
        """The labeled mechanism of each concept."""
        # Get node_labels from the first concept if available
        if (
            self.concepts
            and hasattr(self.concepts[0], "node_labels")
            and self.concepts[0].node_labels is not None
        ):
            label = self.concepts[0].node_labels.indices2labels
            return tuple(list(label(mechanism)) for mechanism in self.mechanisms)
        # Fallback to numeric indices as strings
        return tuple(list(map(str, mechanism)) for mechanism in self.mechanisms)

    def purview_inclusion_of_intersection(self, min_order, max_order):
        return _purview_inclusion(
            "purview_intersection",
            distinctions=self,
            min_order=min_order,
            max_order=max_order,
        )

    def _purview_inclusion_of_union(self, min_order, max_order):
        return _purview_inclusion(
            "purview_union", distinctions=self, min_order=min_order, max_order=max_order
        )

    def purview_inclusion(self, max_order=None):
        """Return a mapping:

        {order: {frozenset[Unit]: {distinctions whose cause/effect purview
                                   union includes those Units}}}
        """
        if max_order is None or max_order not in self._purview_inclusion_by_order:
            self._purview_inclusion_by_order.update(
                # NOTE: We use the union of the cause/effect purviews
                self._purview_inclusion_of_union(
                    min_order=max(self._purview_inclusion_by_order, default=0) + 1,
                    max_order=max_order,
                )
            )
        max_order = fallback(max_order, float("inf"))
        for order, mapping in self._purview_inclusion_by_order.items():
            if order <= max_order:
                yield from mapping.items()

    def resolve_congruence(self, system_state: SystemStateSpecification):
        """Filter out incongruent distinctions."""
        # TODO(4.0) parallelize
        return type(self)(
            filter(
                lambda d: d is not None,
                (distinction.resolve_congruence(system_state) for distinction in self),
            ),
            resolved_congruence=True,
        )

    def mechanism_multiplicities(self):
        return _find_multiplicities(_get_mechanism, self)

    def state_multiplicities(self):
        return _find_multiplicities(_get_state, self)

    @property
    def resolved_congruence(self):
        return self._resolved_congruence


def _null_ces(subsystem=None):  # noqa: ARG001 - subsystem retained for backward-compatible signature
    """Return an empty CES."""
    return CauseEffectStructure(())
