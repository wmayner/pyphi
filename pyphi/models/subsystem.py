#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/subsystem.py

"""Subsystem-level objects."""

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

from toolz import concat

from pyphi.direction import Direction

from .. import utils
from . import cmp, fmt
from .mechanism import Concept

_sia_attributes = ["phi", "ces", "partitioned_ces", "subsystem", "cut_subsystem"]


def _concept_sort_key(concept):
    return (len(concept.mechanism), concept.mechanism)


class CauseEffectStructure(cmp.Orderable, Sequence):
    """A collection of concepts."""

    def __init__(self, concepts=(), subsystem=None, time=None):
        # Normalize the order of concepts
        # TODO(4.0) convert to set?
        self.concepts = tuple(sorted(concepts, key=_concept_sort_key))
        self.subsystem = subsystem
        self.time = time
        self._specifiers = None
        self._purview_inclusion = defaultdict(lambda: 0)
        self._purview_inclusion_max_order = 0

    def __len__(self):
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts)

    def __getitem__(self, i):
        return self.concepts[i]

    def __repr__(self):
        # TODO(4.0) remove dependence on subsystem & time
        return fmt.make_repr(self, ["concepts", "subsystem", "time"])

    def __str__(self):
        return fmt.fmt_ces(self)

    @cmp.sametype
    def __eq__(self, other):
        return self.concepts == other.concepts

    def __hash__(self):
        return hash((self.concepts, self.subsystem))

    def order_by(self):
        return [self.concepts]

    def to_json(self):
        return {
            "concepts": self.concepts,
            "subsystem": self.subsystem,
            "time": self.time,
        }

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

    def purviews(self, direction):
        """Return the purview of each concept in the given direction."""
        for concept in self:
            yield concept.purview(direction)

    @property
    def labeled_mechanisms(self):
        """The labeled mechanism of each concept."""
        # TODO(4.0) remove dependence on subsystem
        label = self.subsystem.node_labels.indices2labels
        return tuple(list(label(mechanism)) for mechanism in self.mechanisms)

    def purview_inclusion(self, degree=None):
        """Map subsets of elements to the number of purviews that include that subset."""
        # TODO(4.0) use lattice datastructure here?
        if degree is None:
            degree = len(self.subsystem)
        degree = min(len(self.subsystem), degree)
        if degree > self._purview_inclusion_max_order:
            for k in range(self._purview_inclusion_max_order + 1, degree + 1):
                for subset in combinations(self.subsystem.node_indices, k):
                    for direction in Direction.both():
                        for purview in self.purviews(direction):
                            if set(subset).issubset(set(purview)):
                                self._purview_inclusion[subset] += 1
            self._purview_inclusion_max_order = degree
        return self._purview_inclusion


class FlatCauseEffectStructure(CauseEffectStructure):
    """A collection of maximally-irreducible components in either causal
    direction."""

    def __init__(self, concepts=(), subsystem=None, time=None):
        if isinstance(concepts, CauseEffectStructure) and not isinstance(
            concepts, FlatCauseEffectStructure
        ):
            subsystem = concepts.subsystem
            time = concepts.time
            concepts = concat((concept.cause, concept.effect) for concept in concepts)
        super().__init__(concepts=concepts, subsystem=subsystem, time=time)

    def __str__(self):
        return fmt.fmt_ces(self, title="Flat cause-effect structure")

    @property
    def purviews(self):
        """The purview of each component."""
        for component in self:
            yield component.purview

    @property
    def specified_purviews(self):
        """The set of unique purviews specified by this CES."""
        return set(self.purviews)

    def specifiers(self, purview):
        """The components that specify the given purview."""
        purview = tuple(purview)
        try:
            return self._specifiers[purview]
        except TypeError:
            self._specifiers = defaultdict(list)
            for component in self:
                self._specifiers[component.purview].append(component)
            return self._specifiers[purview]

    def maximum_specifier(self, purview):
        """Return the components that maximally specify the given purview."""
        return max(self.specifiers(purview))

    def maximum_specifiers(self):
        """Return a mapping from each purview to its maximum specifier."""
        return {purview: self.maximum_specifier(purview) for purview in self.purviews}

    def unflatten(self):
        mechanism_to_mice = defaultdict(dict)
        for mice in self:
            mechanism_to_mice[mice.mechanism][mice.direction] = mice
        return CauseEffectStructure(
            [
                Concept(
                    mechanism=mechanism,
                    cause=mice[Direction.CAUSE],
                    effect=mice[Direction.EFFECT],
                    subsystem=self.subsystem,
                )
                for mechanism, mice in mechanism_to_mice.items()
            ],
            subsystem=self.subsystem,
            time=self.time,
        )

    def purview_occurences(self):
        raise NotImplementedError


class SystemIrreducibilityAnalysis(cmp.Orderable):
    """An analysis of system irreducibility (|big_phi|).

    Contains the |big_phi| value of the |Subsystem|, the cause-effect
    structure, and all the intermediate results obtained in the course of
    computing them.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |big_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the one with the larger subsystem is greater.

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this analysis, *i.e.* the difference between the cause-effect
            structure and the partitioned cause-effect structure for this
            analysis.
        ces (CauseEffectStructure): The cause-effect structure of
            the whole subsystem.
        partitioned_ces (CauseEffectStructure): The cause-effect structure when
            the subsystem is cut.
        subsystem (Subsystem): The subsystem this analysis was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        time (float): The number of seconds it took to calculate.
    """

    def __init__(
        self,
        phi=None,
        ces=None,
        partitioned_ces=None,
        subsystem=None,
        cut_subsystem=None,
        time=None,
    ):
        self.phi = phi
        self.ces = ces
        self.partitioned_ces = partitioned_ces
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
        self.time = time

    def __repr__(self):
        return fmt.make_repr(self, _sia_attributes)

    def __str__(self, ces=True):
        return fmt.fmt_sia(self, ces=ces)

    def print(self, ces=True):
        """Print this |SystemIrreducibilityAnalysis|, optionally without
        cause-effect structures.
        """
        print(self.__str__(ces=ces))

    @property
    def small_phi_time(self):
        """The number of seconds it took to calculate the CES."""
        return self.ces.time

    @property
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem.
        """
        return self.cut_subsystem.cut

    @property
    def network(self):
        """The network the subsystem belongs to."""
        return self.subsystem.network

    unorderable_unless_eq = ["network"]

    def order_by(self):
        return [self.phi, len(self.subsystem), self.subsystem.node_indices]

    def __eq__(self, other):
        return cmp.general_eq(self, other, _sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has
        |big_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.ces,
                self.partitioned_ces,
                self.subsystem,
                self.cut_subsystem,
            )
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            attr: getattr(self, attr)
            for attr in _sia_attributes + ["time", "small_phi_time"]
        }

    @classmethod
    def from_json(cls, dct):
        del dct["small_phi_time"]
        return cls(**dct)


def _null_ces(subsystem):
    """Return an empty CES."""
    ces = CauseEffectStructure((), subsystem=subsystem)
    ces.time = 0.0
    return ces


def _null_sia(subsystem, phi=0.0):
    """Return a |SystemIrreducibilityAnalysis| with zero |big_phi| and empty
    cause-effect structures.

    This is the analysis result for a reducible subsystem.
    """
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        cut_subsystem=subsystem,
        phi=phi,
        ces=_null_ces(subsystem),
        partitioned_ces=_null_ces(subsystem),
    )
