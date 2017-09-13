#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/distance.py

'''
Functions for computing distances between various PyPhi objects.
'''

import numpy as np

from .. import config, utils, validate
from ..distance import big_phi_measure as measure, emd


def concept_distance(c1, c2):
    '''Return the distance between two concepts in concept space.

    Args:
        c1 (Concept): The first concept.
        c2 (Concept): The second concept.

    Returns:
        float: The distance between the two concepts in concept space.
    '''
    # Calculate the sum of the past and future EMDs, expanding the repertoires
    # to the combined purview of the two concepts, so that the EMD signatures
    # are the same size.
    cause_purview = tuple(set(c1.cause.purview + c2.cause.purview))
    effect_purview = tuple(set(c1.effect.purview + c2.effect.purview))
    # Take the sum
    return (measure(c1.expand_cause_repertoire(cause_purview),
                    c2.expand_cause_repertoire(cause_purview)) +
            measure(c1.expand_effect_repertoire(effect_purview),
                    c2.expand_effect_repertoire(effect_purview)))


def _constellation_distance_simple(C1, C2):
    '''Return the distance between two constellations in concept space.

    Assumes the only difference between them is that some concepts have
    disappeared.
    '''
    # Make C1 refer to the bigger constellation.
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    return sum(c.phi * concept_distance(c, c.subsystem.null_concept)
               for c in destroyed)


def _constellation_distance_emd(unique_C1, unique_C2):
    '''Return the distance between two constellations in concept space.

    Uses the generalized EMD.
    '''
    # Get the pairwise distances between the concepts in the unpartitioned and
    # partitioned constellations.
    distances = np.array([
        [concept_distance(i, j) for j in unique_C2] for i in unique_C1
    ])
    # We need distances from all concepts---in both the unpartitioned and
    # partitioned constellations---to the null concept, because:
    # - often a concept in the unpartitioned constellation is destroyed by a
    #   cut (and needs to be moved to the null concept); and
    # - in certain cases, the partitioned system will have *greater* sum of
    #   small-phi, even though it has less big-phi, which means that some
    #   partitioned-constellation concepts will be moved to the null concept.
    distances_to_null = np.array([
        concept_distance(c, c.subsystem.null_concept)
        for constellation in (unique_C1, unique_C2) for c in constellation
    ])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now we make the distance matrix, which will look like this:
    #
    #        C1       C2     0
    #    +~~~~~~~~+~~~~~~~~+~~~+
    #    |        |        |   |
    # C1 |   X    |    D   |   |
    #    |        |        |   |
    #    +~~~~~~~~+~~~~~~~~+ D |
    #    |        |        | n |
    # C2 |   D'   |    X   |   |
    #    |        |        |   |
    #    +~~~~~~~~+~~~~~~~~+~~~|
    #  0 |        Dn'      | X |
    #    +~~~~~~~~~~~~~~~~~~~~~+
    #
    # The diagonal blocks marked with an X are set to a value larger than any
    # pairwise distance between concepts. This ensures that concepts are never
    # moved to another concept within their own constellation; they must always
    # go either from one constellation to another, or to the null concept N.
    # The D block is filled with the pairwise distances between the two
    # constellations, and Dn is filled with the distances from each concept to
    # the null concept.
    N, M = len(unique_C1), len(unique_C2)
    # Add one to the side length for the null concept distances.
    distance_matrix = np.empty([N + M + 1] * 2)
    # Ensure that concepts are never moved within their own constellation.
    distance_matrix[:] = np.max(distances) + 1
    # Set the top-right block to the pairwise constellation distances.
    distance_matrix[:N, N:-1] = distances
    # Set the bottom-left block to the same, but transposed.
    distance_matrix[N:-1, :N] = distances.T
    # Do the same for the distances to the null concept.
    distance_matrix[-1, :-1] = distances_to_null
    distance_matrix[:-1, -1] = distances_to_null.T
    distance_matrix[-1, -1] = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct the two phi distributions, with an entry at the end for the
    # null concept.
    d1 = [c.phi for c in unique_C1] + [0] * M + [0]
    d2 = [0] * N + [c.phi for c in unique_C2] + [0]
    # Calculate how much phi disappeared and assign it to the null concept.
    d2[-1] = sum(d1) - sum(d2)
    # The sum of the two signatures should be the same.
    assert utils.eq(sum(d1), sum(d2))
    # Calculate!
    return emd(np.array(d1), np.array(d2), distance_matrix)


def constellation_distance(C1, C2):
    '''Return the distance between two constellations in concept space.

    Args:
        C1 (Constellation): The first constellation.
        C2 (Constellation): The second constellation.

    Returns:
        float: The distance between the two constellations in concept space.
    '''
    if config.USE_SMALL_PHI_DIFFERENCE_FOR_CONSTELLATION_DISTANCE:
        return round(small_phi_constellation_distance(C1, C2), config.PRECISION)

    concepts_only_in_C1 = [
        c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    concepts_only_in_C2 = [
        c2 for c2 in C2 if not any(c2.emd_eq(c1) for c1 in C1)]
    # If the only difference in the constellations is that some concepts
    # disappeared, then we don't need to use the EMD.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        dist = _constellation_distance_simple(C1, C2)
    else:
        dist = _constellation_distance_emd(concepts_only_in_C1,
                                           concepts_only_in_C2)

    return round(dist, config.PRECISION)


def small_phi_constellation_distance(C1, C2):
    '''Return the difference in |small_phi| between constellations.'''
    return sum(c.phi for c in C1) - sum(c.phi for c in C2)
