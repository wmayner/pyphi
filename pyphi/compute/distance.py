#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/distance.py

"""
Functions for computing distances between various PyPhi objects.
"""

import numpy as np

from .. import config, utils
from ..distance import emd
from ..distance import system_repertoire_distance as repertoire_distance


def concept_distance(c1, c2):
    """Return the distance between two concepts in concept space.

    Args:
        c1 (Concept): The first concept.
        c2 (Concept): The second concept.

    Returns:
        float: The distance between the two concepts in concept space.
    """
    # Calculate the sum of the cause and effect EMDs, expanding the repertoires
    # to the combined purview of the two concepts, so that the EMD signatures
    # are the same size.
    cause_purview = tuple(set(c1.cause.purview + c2.cause.purview))
    effect_purview = tuple(set(c1.effect.purview + c2.effect.purview))
    # Take the sum
    return (repertoire_distance(c1.expand_cause_repertoire(cause_purview),
                                c2.expand_cause_repertoire(cause_purview)) +
            repertoire_distance(c1.expand_effect_repertoire(effect_purview),
                                c2.expand_effect_repertoire(effect_purview)))


def _ces_distance_simple(C1, C2):
    """Return the distance between two cause-effect structures.

    Assumes the only difference between them is that some concepts have
    disappeared.
    """
    # Make C1 refer to the bigger CES.
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    return sum(c.phi * concept_distance(c, c.subsystem.null_concept)
               for c in destroyed)


def _ces_distance_emd(unique_C1, unique_C2):
    """Return the distance between two cause-effect structures.

    Uses the generalized EMD.
    """
    # Get the pairwise distances between the concepts in the unpartitioned and
    # partitioned CESs.
    distances = np.array([
        [concept_distance(i, j) for j in unique_C2] for i in unique_C1
    ])
    # We need distances from all concepts---in both the unpartitioned and
    # partitioned CESs---to the null concept, because:
    # - often a concept in the unpartitioned CES is destroyed by a
    #   cut (and needs to be moved to the null concept); and
    # - in certain cases, the partitioned system will have *greater* sum of
    #   small-phi, even though it has less big-phi, which means that some
    #   partitioned-CES concepts will be moved to the null concept.
    distances_to_null = np.array([
        concept_distance(c, c.subsystem.null_concept)
        for ces in (unique_C1, unique_C2) for c in ces
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
    # moved to another concept within their own CES; they must always go either
    # from one CES to another, or to the null concept N. The D block is filled
    # with the pairwise distances between the two CESs, and Dn is filled with
    # the distances from each concept to the null concept.
    N, M = len(unique_C1), len(unique_C2)
    # Add one to the side length for the null concept distances.
    distance_matrix = np.empty([N + M + 1] * 2)
    # Ensure that concepts are never moved within their own CES.
    distance_matrix[:] = np.max(distances) + 1
    # Set the top-right block to the pairwise CES distances.
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


def ces_distance(C1, C2):
    """Return the distance between two cause-effect structures.

    Args:
        C1 (CauseEffectStructure): The first |CauseEffectStructure|.
        C2 (CauseEffectStructure): The second |CauseEffectStructure|.

    Returns:
        float: The distance between the two cause-effect structures in concept
        space.
    """
    if config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE:
        return round(small_phi_ces_distance(C1, C2), config.PRECISION)

    concepts_only_in_C1 = [
        c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    concepts_only_in_C2 = [
        c2 for c2 in C2 if not any(c2.emd_eq(c1) for c1 in C1)]
    # If the only difference in the CESs is that some concepts
    # disappeared, then we don't need to use the EMD.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        dist = _ces_distance_simple(C1, C2)
    else:
        dist = _ces_distance_emd(concepts_only_in_C1, concepts_only_in_C2)

    return round(dist, config.PRECISION)


def small_phi_ces_distance(C1, C2):
    """Return the difference in |small_phi| between |CauseEffectStructure|."""
    return sum(c.phi for c in C1) - sum(c.phi for c in C2)
