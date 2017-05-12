#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/concept.py

import multiprocessing
from time import time

from . import parallel
from .distance import constellation_distance
from .. import config, models, utils


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


class ComputeConstellation(parallel.MapReduce):
    """Engine for computing a constellation."""
    description = 'Computing concepts'

    def empty_result(self, *args):
        return []

    def compute(self, mechanism, subsystem, purviews, past_purviews,
                future_purviews):
        """Compute a concept for a mechanism, in this subsystem with the
        provided purviews."""
        return concept(subsystem, mechanism, purviews=purviews,
                       past_purviews=past_purviews,
                       future_purviews=future_purviews)

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero phi to the constellation."""
        if new_concept.phi > 0:
            concepts.append(new_concept)
        return concepts


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
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)

    engine = ComputeConstellation(mechanisms, subsystem, purviews,
                                  past_purviews, future_purviews)
    return models.Constellation(engine.run(config.PARALLEL_CONCEPT_EVALUATION))


def conceptual_information(subsystem):
    """Return the conceptual information for a subsystem.

    This is the distance from the subsystem's constellation to the null
    concept.
    """
    ci = constellation_distance(constellation(subsystem), ())
    return round(ci, config.PRECISION)
