#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/concept.py

"""
Functions for computing concepts and constellations of concepts.
"""

import multiprocessing
from time import time

from . import parallel
from .. import config, models, utils
from .distance import constellation_distance


def concept(subsystem, mechanism, purviews=False, past_purviews=False,
            future_purviews=False):
    """Return the concept specified by a mechanism within a subsytem.

    Args:
        subsystem (Subsytem): The context in which the mechanism should be
            considered.
        mechanism (tuple[int]): The candidate set of nodes.

    Keyword Args:
        purviews (tuple[tuple[int]]): Restrict the possible purviews to those
            in this list.
        past_purviews (tuple[tuple[int]]): Restrict the possible cause
            purviews to those in this list. Takes precedence over ``purviews``.
        future_purviews (tuple[tuple[int]]): Restrict the possible effect
            purviews to those in this list. Takes precedence over ``purviews``.

    Returns:
        |Concept|: The pair of maximally irreducible cause/effect repertoires
        that constitute the concept specified by the given mechanism.
    """
    start = time()

    # If the mechanism is empty, there is no concept.
    if not mechanism:
        concept = subsystem.null_concept
    else:
        concept = subsystem.concept(
            mechanism, purviews=purviews, past_purviews=past_purviews,
            future_purviews=future_purviews)

    concept.time = round(time() - start, config.PRECISION)
    return concept


def _sequential_constellation(subsystem, mechanisms, purviews=False,
                              past_purviews=False, future_purviews=False):
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


def _parallel_constellation(subsystem, mechanisms, purviews=False,
                            past_purviews=False, future_purviews=False):

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
        mechanisms (tuple[tuple[int]]): A list of mechanisms, as node indices,
            to be considered as possible mechanisms for the concepts in the
            constellation.
        purviews (tuple[tuple[int]]): A list of purviews, as node indices, to
            be considered as possible purviews for the concepts in the
            constellation.
        past_purviews (tuple[tuple[int]]): A list of purviews, as node indices,
            to be considered as possible *cause* purviews for the concepts in
            the constellation. This takes precedence over the more general
            ``purviews`` option.
        future_purviews (tuple[tuple[int]]): A list of purviews, as node
            indices, to be considered as possible *effect* purviews for the
            concepts in the constellation. This takes precedence over the more
            general ``purviews`` option.

    Returns:
        |Constellation|: A tuple of every |Concept| in the constellation.
    """

    if config.PARALLEL_CONCEPT_EVALUATION:
        constellation = _parallel_constellation
    else:
        constellation = _sequential_constellation

    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)

    return constellation(subsystem, mechanisms, purviews, past_purviews,
                         future_purviews)


def conceptual_information(subsystem):
    """Return the conceptual information for a subsystem.

    This is the distance from the subsystem's constellation to the null
    concept.
    """
    ci = constellation_distance(constellation(subsystem), ())
    return round(ci, config.PRECISION)
