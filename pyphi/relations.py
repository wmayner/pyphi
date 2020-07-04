#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relations.py

"""Functions for computing relations between concepts."""

import operator
from itertools import product

import numpy as np
from toolz import concat, curry

from . import config, validate
from .models import cmp
from .models.cuts import Bipartition, Part
from .models.subsystem import CauseEffectStructure
from .utils import powerset, eq

# TODO there should be an option to resolve ties at different levels

# TODO Requests from Matteo
# - have relations report their type (supertext, etc.; ask andrew/matteo)
# - consider refactoring subsystem reference off the concepts for serialization
# - node labelzzzzz!

# TODO notes from Matteo 2019-02-20
# - object encapsulating interface to pickled concepts from CHTC for matteo and andrew


@curry
def _all_same(comparison, seq):
    sentinel = object()
    first = next(seq, sentinel)
    if first is sentinel:
        # Vacuously
        return True
    return all(comparison(first, other) for other in seq)


# Compare equality up to precision
all_are_equal = _all_same(eq)
all_are_identical = _all_same(operator.is_)


# TODO test
@curry
def _all_extrema(comparison, seq):
    """Return the extrema of ``seq``.

    Use ``<`` as the comparison to obtain the minima; use ``>`` as the
    comparison to obtain the maxima.

    Uses only one pass through ``seq``.

    Args:
        comparison (callable): A comparison operator.
        seq (iterator): An iterator over a sequence.

    Returns:
        list: The maxima/minima in ``seq``.
    """
    extrema = []
    sentinel = object()
    current_extremum = next(seq, sentinel)
    if current_extremum is sentinel:
        # Return an empty list if the sequence is empty
        return extrema
    extrema.append(current_extremum)
    for element in seq:
        if comparison(element, current_extremum):
            extrema = [element]
            current_extremum = element
        elif element == current_extremum:
            extrema.append(element)
    return extrema


all_minima = _all_extrema(operator.lt)
all_maxima = _all_extrema(operator.gt)


def indices(iterable):
    """Convert an iterable to element indices."""
    return tuple(sorted(iterable))


# TODO rename and hook into config options
def divergence(p, q):
    # We don't care if p or q or both are zero
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(p * np.nan_to_num(np.log2(p / q)))


# TODO this should end up being stored on the MICE object itself when the 4.0
# branch is finished
def maximal_state(mice):
    """Return the maximally divergent state(s) for this MICE.

    Note that there can be ties.

    Returns:
        np.array: A 2D array where each row is a maximally divergent state.
    """
    div = divergence(mice.repertoire, mice.partitioned_repertoire)
    return np.transpose(np.where(div == div.max()))


def congruent_nodes(states):
    """Return the set of nodes that have the same state in all given states."""
    return set(np.all(states == states[0], axis=0).nonzero()[0])


class Relation(cmp.Orderable):
    """A relation among causes/effects."""

    def __init__(self, relata, purview, phi, ties=None):
        self._relata = relata
        self._purview = indices(purview)
        self._phi = phi
        self._ties = ties

    @property
    def relata(self):
        return self._relata

    @property
    def purview(self):
        return self._purview

    @property
    def phi(self):
        return self._phi

    @property
    def ties(self):
        return self._ties

    @property
    def subsystem(self):
        return self.relata.subsystem

    @property
    def mechanisms(self):
        return [relatum.mechanism for relatum in self.relata]

    def __repr__(self):
        return f"Relation({self.mechanisms}, {self.purview}, {self.phi})"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        return bool(round(self.phi, config.PRECISION) > 0.0)

    def __eq__(self, other):
        attrs = ["phi", "relata"]
        return cmp.general_eq(self, other, attrs)

    def order_by(self):
        # TODO check with andrew/matteo about this; what do we want? maybe also
        # the order?
        return (round(self.phi, config.PRECISION), len(self.relata))

    @staticmethod
    def union(tied_relations):
        """Return the 'union' of tied relations.

        This is a new Relation object that contains the purviews of the other
        relations in the ``ties`` attribute.
        """
        if not tied_relations:
            raise ValueError("tied relations cannot be empty")
        if not all_are_equal(r.phi for r in tied_relations):
            raise ValueError("tied relations must have the same phi")
        if not all_are_identical(r.relata for r in tied_relations):
            raise ValueError("tied relations must be among the same relata.")
        first = tied_relations[0]
        tied_purviews = set(r.purview for r in tied_relations)
        return Relation(first.relata, first.purview, first.phi, ties=tied_purviews)


class Relata:
    """A set of potentially-related causes/effects."""

    def __init__(self, subsystem, relata):
        validate.relata(relata)
        # TODO do we want to use sorted() here to ensure equality comparisons
        # are correct?
        self._relata = relata
        self._subsystem = subsystem
        self._maximal_states = None

    @property
    def subsystem(self):
        return self._subsystem

    @property
    def mechanisms(self):
        return (relatum.mechanism for relatum in self)

    @property
    def purviews(self):
        return (relatum.purview for relatum in self)

    # TODO !!! remove once the maximal states are on the MICE objects
    @property
    def maximal_states(self):
        if self._maximal_states is None:
            self._maximal_states = {mice: maximal_state(mice) for mice in self}
        return self._maximal_states

    def __repr__(self):
        mechanisms = list(self.mechanisms)
        purviews = list(self.purviews)
        return f"Relata(mechanisms={mechanisms}, purviews={purviews})"

    def __str__(self):
        return repr(self)

    def __iter__(self):
        """Iterate over relata."""
        return iter(self._relata)

    def __getitem__(self, index):
        return self._relata[index]

    def __len__(self):
        return len(self._relata)

    # TODO this relies on the implementation of equality for MICEs; we should
    # go back and make sure that implementation is still appropriate
    def __eq__(self, other):
        return all(mice == other_mice for mice, other_mice in zip(self, other))

    def overlap(self):
        """Return the set of elements that are in the purview of every relatum."""
        return set.intersection(*map(set, self.purviews))

    def null_relation(self, purview=None, phi=0.0):
        if purview is None:
            purview = set()
        return Relation(self._relata, purview, phi)

    def congruent_overlap(self):
        """Yield the congruent overlap(s) among the relata.

        These are the common purview elements among the relata whose
        maximally-divergent states are consistent; that is, the largest subset
        of the union of the purviews such that, for each element, that
        element's state is the same according to the maximally divergent state
        of each relatum.

        Note that there can be multiple congruent overlaps.
        """
        overlap = self.overlap()
        # A state set is one state per relatum; a relatum can have multiple
        # tied states, so we consider every combination
        for state_set in product(*self.maximal_states.values()):
            # Get the nodes that have the same state in every maximal state
            congruent = congruent_nodes(state_set)
            # Find the largest congruent subset of the full overlap
            intersection = set.intersection(overlap, congruent)
            if intersection:
                yield intersection

    def possible_purviews(self):
        """Return all possible purviews.

        This is the powerset of the congruent overlap. If there are multiple
        congruent overlaps because of ties, it is the union of the powerset of
        each.
        """
        # TODO note: ties are included here
        return map(
            set,
            concat(
                powerset(overlap, nonempty=True) for overlap in self.congruent_overlap()
            ),
        )

    def partitioned_divergence(self, purview, mice):
        """Return the maximal partitioned divergence over this purview.

        The purview is cut away from the MICE and the divergence is computed
        between the unpartitioned repertoire and partitioned repertoire.

        If the MICE has multiple tied maximally-divergent states, we take the
        maximum unpartitioned-partitioned divergence across those tied states.

        Args:
            purview (set): The set of node indices in the purview.
            mice (|MICE|): The |MICE| object to consider.
        """
        non_purview_indices = tuple(set(mice.purview) - purview)
        partition = Bipartition(
            Part(mice.mechanism, non_purview_indices), Part((), tuple(purview))
        )
        partitioned_repertoire = self.subsystem.partitioned_repertoire(
            mice.direction, partition
        )
        div = divergence(mice.repertoire, partitioned_repertoire)
        state_indices = tuple(np.transpose(self.maximal_states[mice]))
        # TODO tie breaking happens here! double-check with andrew
        return np.max(div[state_indices])

    # TODO: do we care about ties here?
    # 2019-05-30: no, according to andrew
    def minimum_information_relation(self, purview):
        """Return the minimal information relation for this purview.

        Args:
            relata (Relata): The relata to consider.
            purview (set): The purview to consider.
        """
        phi_min = float("inf")
        for mice in self:
            phi = self.partitioned_divergence(purview, mice)
            # Short circuit if phi is zero
            if phi == 0.0:
                return self.null_relation(purview=purview, phi=0.0)
            # Update the minimal relation if phi is smaller
            if phi < phi_min:
                phi_min = phi
        return Relation(self, purview, phi_min)

    def maximally_irreducible_relation(self):
        """Return the maximally-irreducible relation among these relata.

        If there are ties, the tied relations will be recorded in the 'ties'
        attribute of the returned relation.

        Returns:
            Relation: the maximally irreducible relation among these relata,
            with any tied purviews recorded.
        """
        if len(self) == 1:
            # Singletons cannot have relations
            return self.null_relation()
        # Find maximal relations
        tied_relations = all_maxima(
            map(self.minimum_information_relation, self.possible_purviews())
        )
        if not tied_relations:
            return self.null_relation()
        # Keep track of ties
        return Relation.union(tied_relations)


def relation(relata):
    """Return the maximally irreducible relation among the given relata.

    Alias for the ``Relata.maximally_irreducible_relation()`` method.
    """
    return relata.maximally_irreducible_relation()


def separate_ces(ces):
    """Return the individual causes and effects, unpaired, from a CES."""
    return CauseEffectStructure(
        concat((concept.cause, concept.effect) for concept in ces)
    )


# TODO add order kwarg to restrict to just a certain order
# TODO: change to candidate_relations?
def all_relations(subsystem, ces):
    """Return all relations, even those with zero phi."""
    # Relations can be over any combination of causes/effects in the CES, so we
    # get a flat list of all causes and effects
    ces = separate_ces(ces)
    # Compute all relations
    return map(
        relation, (Relata(subsystem, subset) for subset in filter(
            lambda purviews: len(purviews) > 1,
            powerset(ces, nonempty=True))
        )
    )


def relations(subsystem, ces):
    """Return the irreducible relations among the causes/effects in the CES."""
    return filter(None, all_relations(subsystem, ces))
