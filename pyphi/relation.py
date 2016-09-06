#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relation.py

import numpy as np

from . import models, utils, validate


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

    def purview_overlap(self, direction):
        """All elements in the purview of every concept in the set."""
        return indices(set.intersection(*[set(c.purview(direction))
                                          for c in self.concepts]))

    def possible_purviews(self, direction):
        """Possible purviews of this set of concepts."""
        return list(utils.powerset(self.purview_overlap(direction)))[1:]


def relation_partition(direction, concept, purview):
    """Cut all connections to or from the purview, depending on direction."""
    p0m = concept.mechanism
    p0p = tuple(set(concept.purview(direction)) - set(purview))
    part0 = models.Part(p0m, p0p)
    part1 = models.Part((), purview)

    return models.Bipartition(part0, part1)


def find_relation(direction, concept_list):

    concept_set = ConceptSet(concept_list)
    subsystem = concept_set.subsystem

    max_purview = None
    max_purview_phi = 0

    for purview in concept_set.possible_purviews(direction):

        min_phi = float('inf')

        for concept in concept_set:
            # Recompute the concept
            # TODO: clarify that this is correct - or do we compute the
            # entire Concept? or the Mice?

            partition = relation_partition(direction, concept, purview)

            partitioned_repertoire = subsystem.partitioned_repertoire(
                direction, partition)

            phi = utils.hamming_emd(concept.repertoire(direction),
                                    partitioned_repertoire)

            print(direction, phi)
            if phi < min_phi:
                min_phi = phi

        if min_phi > max_purview_phi:
            max_purview_phi = min_phi
            max_purview = purview

    return max_purview, max_purview_phi
