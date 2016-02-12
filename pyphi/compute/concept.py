#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/concept.py

import multiprocessing
from time import time

import numpy as np

from . import parallel
from .. import config, models, utils


def concept(subsystem, mechanism, purviews=False, past_purviews=False,
            future_purviews=False):
    """Return the concept specified by a mechanism within a subsytem.

    Args:
        subsystem (Subsytem): The context in which the mechanism should be
            considered.
        mechanism (tuple(int)): The candidate set of nodes.

    Keyword Args:
        purviews (tuple(tuple(int))): Restrict the possible purviews to those
            in this list.
        past_purviews (tuple(tuple(int))): Restrict the possible cause
            purviews to those in this list. Takes precedence over ``purviews``.
        future_purviews (tuple(tuple(int))): Restrict the possible effect
            purviews to those in this list. Takes precedence over ``purviews``.

    Returns:
        concept (|Concept|): The pair of maximally irreducible cause/effect
            repertoires that constitute the concept specified by the given
            mechanism.
    """
    start = time()

    # If the mechanism is empty, there is no concept.
    if not mechanism:
        concept = subsystem.null_concept
    else:
        concept = subsystem.concept(
            mechanism, purviews=purviews, past_purviews=past_purviews,
            future_purviews=future_purviews)

    concept.time = time() - start
    return concept


def _sequential_constellation(subsystem, mechanisms=False, purviews=False,
                              past_purviews=False, future_purviews=False):
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)
    concepts = [concept(subsystem, mechanism, purviews=purviews,
                        past_purviews=past_purviews,
                        future_purviews=future_purviews)
                for mechanism in mechanisms]
    # Filter out falsy concepts, i.e. those with effectively zero Phi.
    return models.Constellation(filter(None, concepts))


def _concept_wrapper(in_queue, out_queue, subsystem, purviews=False,
                     past_purviews=False, future_purviews=False):
    """Wrapper for parallel evaluation of concepts."""
    while True:
        mechanism = in_queue.get()
        if mechanism is None:
            break
        new_concept = concept(subsystem, mechanism, purviews=purviews,
                              past_purviews=past_purviews,
                              future_purviews=future_purviews)
        if new_concept.phi > 0:
            out_queue.put(new_concept)
    out_queue.put(None)


def _parallel_constellation(subsystem, mechanisms=False, purviews=False,
                            past_purviews=False, future_purviews=False):
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)

    number_of_processes = parallel.get_num_processes()

    # Define input and output queues and load the input queue with all possible
    # cuts and a 'poison pill' for each process.
    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    for mechanism in mechanisms:
        in_queue.put(mechanism)
    for i in range(number_of_processes):
        in_queue.put(None)

    # Initialize the processes and start them.
    for i in range(number_of_processes):
        args = (in_queue, out_queue, subsystem, purviews,
                past_purviews, future_purviews)
        process = multiprocessing.Process(target=_concept_wrapper, args=args)
        process.start()

    # Continue to process output queue until all processes have completed, or a
    # 'poison pill' has been returned.
    concepts = []
    while True:
        new_concept = out_queue.get()
        if new_concept is None:
            number_of_processes -= 1
            if number_of_processes == 0:
                break
        else:
            concepts.append(new_concept)
    return models.Constellation(concepts)


def constellation(subsystem, mechanisms=False, purviews=False,
                  past_purviews=False, future_purviews=False):
    """Return the conceptual structure of this subsystem, optionally restricted
    to concepts with the mechanisms and purviews given in keyword arguments.

    If you will not be using the full constellation, restricting the possible
    mechanisms and purviews can make this function much faster.

    Args:
        subsystem (Subsystem): The subsystem for which to determine the
            constellation.

    Keyword Args:
        mechanisms (tuple(tuple(int))): A list of mechanisms, as node indices,
            to be considered as possible mechanisms for the concepts in the
            constellation.
        purviews (tuple(tuple(int))): A list of purviews, as node indices, to
            be considered as possible purviews for the concepts in the
            constellation.
        past_purviews (tuple(tuple(int))): A list of purviews, as node indices,
            to be considered as possible *cause* purviews for the concepts in
            the constellation. This takes precedence over the more general
            ``purviews`` option.
        future_purviews (tuple(tuple(int))): A list of purviews, as node
            indices, to be considered as possible *effect* purviews for the
            concepts in the constellation. This takes precedence over the more
            general ``purviews`` option.

    Returns:
        constellation (Constellation): A tuple of all the Concepts in the
            constellation.
    """

    if config.PARALLEL_CONCEPT_EVALUATION:
        constellation = _parallel_constellation
    else:
        constellation = _sequential_constellation

    return constellation(subsystem, mechanisms, purviews, past_purviews,
                         future_purviews)


def concept_distance(c1, c2):
    """Return the distance between two concepts in concept-space.

    Args:
        c1 (Mice): The first concept.
        c2 (Mice): The second concept.

    Returns:
        distance (``float``): The distance between the two concepts in
            concept-space.
    """
    # Calculate the sum of the past and future EMDs, expanding the repertoires
    # to the combined purview of the two concepts, so that the EMD signatures
    # are the same size.
    cause_purview = tuple(set(c1.cause.purview + c2.cause.purview))
    effect_purview = tuple(set(c1.effect.purview + c2.effect.purview))
    return sum([
        utils.hamming_emd(c1.expand_cause_repertoire(cause_purview),
                          c2.expand_cause_repertoire(cause_purview)),
        utils.hamming_emd(c1.expand_effect_repertoire(effect_purview),
                          c2.expand_effect_repertoire(effect_purview))])


def _constellation_distance_simple(C1, C2):
    """Return the distance between two constellations in concept-space.

    Assumes the only difference between them is that some concepts have
    disappeared.
    """
    # Make C1 refer to the bigger constellation.
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    return sum(c.phi * concept_distance(c, c.subsystem.null_concept)
               for c in destroyed)


def _constellation_distance_emd(unique_C1, unique_C2):
    """Return the distance between two constellations in concept-space.

    Uses the generalized EMD.
    """
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
    assert utils.phi_eq(sum(d1), sum(d2))
    # Calculate!
    return utils.emd(np.array(d1), np.array(d2), distance_matrix)


def constellation_distance(C1, C2):
    """Return the distance between two constellations in concept-space.

    Args:
        C1 (Constellation): The first constellation.
        C2 (Constellation): The second constellation.

    Returns:
        distance (``float``): The distance between the two constellations in
            concept-space.
    """
    concepts_only_in_C1 = [
        c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    concepts_only_in_C2 = [
        c2 for c2 in C2 if not any(c2.emd_eq(c1) for c1 in C1)]
    # If the only difference in the constellations is that some concepts
    # disappeared, then we don't need to use the EMD.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        return _constellation_distance_simple(C1, C2)
    else:
        return _constellation_distance_emd(concepts_only_in_C1,
                                           concepts_only_in_C2)


def conceptual_information(subsystem):
    """Return the conceptual information for a subsystem.

    This is the distance from the subsystem's constellation to the null
    concept.
    """
    ci = constellation_distance(constellation(subsystem), ())
    return round(ci, config.PRECISION)
