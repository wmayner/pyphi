#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relation.py

import numpy as np

from . import models, utils, validate
from .subsystem import emd


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
        return utils.powerset(self.purview_overlap(direction),
                              include_empty=False)


def relation_partitions(direction, concept, purview):
    """Yield a generator over all possible partitions of the purview."""
    for purview_subset in utils.powerset(purview, include_empty=False):
        p0m = concept.mechanism
        p0p = tuple(set(concept.purview(direction)) - set(purview_subset))
        part0 = models.Part(p0m, p0p)
        part1 = models.Part((), purview_subset)

        yield models.Bipartition(part0, part1)


def find_relation(direction, concept_list):
    """Find the relation between a set of concepts.

    Arguments:
        direction (str): |past| or |future|.
        concept_list (tuple[Concept]): A list of concepts.

    Returns:
        MaximalOverlap: The maximally irreducible overlap of the concepts.
    """

    concept_set = ConceptSet(concept_list)
    subsystem = concept_set.subsystem

    max_purview = None
    max_purview_phi = 0.0

    for purview in concept_set.possible_purviews(direction):

        min_phi = float('inf')

        for concept in concept_set:
            # Cut inputs/outputs of purview and recompute the concept
            repertoire = concept.repertoire(direction)

            def partition_distance(partition):
                partitioned_repertoire = subsystem.partitioned_repertoire(
                    direction, partition)
                return emd(direction, repertoire, partitioned_repertoire)

            phi = min(partition_distance(partition)
                      for partition in relation_partitions(
                        direction, concept, purview))

            if phi < min_phi:
                min_phi = phi

        if min_phi > max_purview_phi:
            max_purview_phi = min_phi
            max_purview = purview

    return MaximalOverlap(max_purview_phi, max_purview, direction, concept_set)


# TODO: rename?
class MaximalOverlap:
    """A maximally irreducible overlap.

    Defines the relationship between a set of concepts

    Attributes:
        phi (float): The measure of the irreducibility of these concepts.
        purview (tuple(int)): The maximally irreducible overlap.
        direction (str): |past| or |future|.
        concepts (ConceptSet): The concepts over which this overlap is
            computed.
    """

    def __init__(self, phi, purview, direction, concepts):
        self.phi = phi
        self.purview = purview
        self.direction = direction
        self.concepts = concepts

    def __repr__(self):
        attrs = ['phi', 'purview', 'direction', 'concepts']
        return models.fmt.make_repr(self, attrs)

    def __str__(self):
        return "MaximalOverlap(phi={}, purview={}, direction={})".format(
            self.phi, self.purview, self.direction)
