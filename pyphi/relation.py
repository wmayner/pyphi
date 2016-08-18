#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relation.py

from collections import namedtuple

import numpy as np

import pyphi
from . import utils
from .constants import DIRECTIONS, PAST, FUTURE


def indices(iterable):
    """Convert an iterable to element indices."""
    return tuple(sorted(iterable))


class ConceptSet:
    """A set of concepts to be evaluated for a relationship"""

    def __init__(self, concepts):
        if len(concepts) == 0:
            raise ValueError('ConceptSet cannot be empty')

        self.concepts = concepts

    @property
    def subsystem(self):
        subsystem = self.concepts[0].subsystem
        assert (subsystem == c.subsystem for c in self.concepts)
        return subsystem

    def shared_purview(self, direction):
        """All elements in the purview of every concept in the set."""

        if direction == DIRECTIONS[PAST]:
            attr = 'cause_purview'
        elif direction == DIRECTIONS[FUTURE]:
            attr = 'effect_purview'

        return indices(set.intersection(*[set(getattr(c, attr))
                                          for c in self.concepts]))

    def possible_purviews(self, direction):
        """Possible purviews of this set of concepts."""
        return list(utils.powerset(self.shared_purview(direction)))[1:]


class RelationCut(namedtuple('RelationCut',
                             ['direction', 'purview', 'non_purview'])):
    """Cut purview elements.

    The outputs of the purview are cut if we are computing causes, the inputs
    of the purview are cut if we are computing effects.
    """
    @property
    def indices(self):
        """The indices of the cut."""
        return indices(self.purview + self.non_purview)

    def apply_cut(self, cm):
        cm = cm.copy()

        # Cut outputs of purview
        if self.direction == DIRECTIONS[PAST]:
            cm[self.purview, :] = 0

        # Cut inputs of purview
        elif self.direction == DIRECTIONS[FUTURE]:
            cm[:, self.purview] = 0

        return cm

    def cut_matrix(self):
        return "NO MATRIX"


def cut_subsystem(direction, purview, concept_set):
    """Cut a subsystem by severing the purview (joint constraint)."""
    subsystem = concept_set.subsystem
    non_purview = indices(set(subsystem.node_indices) - set(purview))

    cut = RelationCut(direction, purview, non_purview)

    # TODO: is this safe, given the Mice cache is reused?
    return subsystem.apply_cut(cut)


def find_relation(direction, concept_list):
    """
    for purview:
        cut_system = cut_inputs_or_outputs(direction, purview, subsystem)

        for concept in concepts:
            cut_concept = recompute_concept(cut_system concept)

            phi_diff = concept.phi - cut_concept.phi

        min_phi_diff = min(concept.phi_diff)

    maximally_irreducible_overlap = max(purview.min_phi_diff) # "small phi"
    """
