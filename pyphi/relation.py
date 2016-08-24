#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relation.py

from collections import namedtuple

import numpy as np

import pyphi
from . import utils, validate
from .constants import DIRECTIONS, PAST, FUTURE


def indices(iterable):
    """Convert an iterable to element indices."""
    return tuple(sorted(iterable))


class ConceptSet:
    """A set of concepts to be evaluated for a relationship"""

    def __init__(self, concepts):
        self.concepts = concepts

        validate.concept_set(concepts)

    def __iter__(self):
        """Iterate over concepts in the set."""
        return iter(self.concepts)

    @property
    def subsystem(self):
        return self.concepts[0].subsystem

    def shared_purview(self, direction):
        """All elements in the purview of every concept in the set."""
        return indices(set.intersection(*[set(c.purview(direction))
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

    # TODO: implement so that we can reuse the mice cache
    def cut_matrix(self):
        return "NO MATRIX"


def cut_subsystem(direction, purview, concept_set):
    """Cut a subsystem by severing the purview (joint constraint)."""
    subsystem = concept_set.subsystem
    non_purview = indices(set(subsystem.node_indices) - set(purview))

    cut = RelationCut(direction, purview, non_purview)

    # TODO: implement `cut.cut_matrix` so that we can reuse the cache
    return subsystem.apply_cut(cut, _reuse_cache=False)


def find_relation(direction, concept_list):

    concept_set = ConceptSet(concept_list)

    max_purview = None
    max_purview_phi = 0

    for purview in concept_set.possible_purviews(direction):

        cut_system = cut_subsystem(direction, purview, concept_set)

        min_phi_diff = float('inf')

        for concept in concept_set:
            # Recompute the concept
            # TODO: clarify that this is correct - or do we compute the
            # entire Concept? or the Mice?
            cut_phi = cut_system.find_mip(direction, concept.mechanism,
                                          concept.purview(direction)).phi
            phi_diff = concept.phi - cut_phi
            if phi_diff < min_phi_diff:
                min_phi_diff = phi_diff

        if min_phi_diff > max_purview_phi:
            max_purview_phi = min_phi_diff
            max_purview = purview

    return max_purview, max_purview_phi
