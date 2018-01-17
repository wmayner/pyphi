#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/subsystem.py

"""Subsystem-level objects."""

from . import cmp, fmt
from .. import config, utils

# pylint: disable=too-many-arguments

_sia_attributes = ['phi', 'ces', 'partitioned_ces', 'subsystem',
                   'cut_subsystem']


def _concept_sort_key(concept):
    return (len(concept.mechanism), concept.mechanism)


class CauseEffectStructure(tuple):
    """A collection of concepts."""
    # TODO: compare CESs using set equality

    def __new__(cls, concepts=()):
        """Normalize the order of concepts in the |CauseEffectStructure|."""
        return super().__new__(cls, sorted(concepts, key=_concept_sort_key))

    def __repr__(self):
        if config.REPR_VERBOSITY > 0:
            return self.__str__()

        return "CauseEffectStructure{}".format(
            super().__repr__())

    def __str__(self):
        return fmt.fmt_ces(self)

    def to_json(self):
        return {'concepts': list(self)}

    @property
    def mechanisms(self):
        """The mechanism of each concept."""
        return [concept.mechanism for concept in self]

    @property
    def phis(self):
        """The |small_phi| values of each concept."""
        return [concept.phi for concept in self]

    @property
    def labeled_mechanisms(self):
        """The labeled mechanism of each concept."""
        if not self:
            return []
        label = self[0].subsystem.node_labels.indices2labels
        return tuple(list(label(mechanism)) for mechanism in self.mechanisms)

    @classmethod
    def from_json(cls, json):
        return cls(json['concepts'])


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
        small_phi_time (float): The number of seconds it took to calculate the
            cause-effect structure.
    """

    def __init__(self, phi=None, ces=None,
                 partitioned_ces=None, subsystem=None,
                 cut_subsystem=None, time=None, small_phi_time=None):
        self.phi = phi
        self.ces = ces
        self.partitioned_ces = partitioned_ces
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
        self.time = time
        self.small_phi_time = small_phi_time

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
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem.
        """
        return self.cut_subsystem.cut

    @property
    def network(self):
        """The network the subsystem belongs to."""
        return self.subsystem.network

    unorderable_unless_eq = ['network']

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
        return hash((self.phi,
                     self.ces,
                     self.partitioned_ces,
                     self.subsystem,
                     self.cut_subsystem))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            attr: getattr(self, attr)
            for attr in _sia_attributes + ['time', 'small_phi_time']
        }


def _null_sia(subsystem, phi=0.0):
    """Return a |SystemIrreducibilityAnalysis| with zero |big_phi| and empty
    cause-effect structures.

    This is the analysis result for a reducible subsystem.
    """
    return SystemIrreducibilityAnalysis(subsystem=subsystem,
                                        cut_subsystem=subsystem,
                                        phi=phi,
                                        ces=(),
                                        partitioned_ces=())
