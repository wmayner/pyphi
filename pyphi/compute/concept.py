#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/concept.py

"""
Functions for computing concepts and cause-effect structures.
"""

# pylint: disable=too-many-arguments,redefined-outer-name

import logging
from time import time

from . import parallel
from .. import config, models, utils
from .distance import ces_distance

log = logging.getLogger(__name__)


def concept(subsystem, mechanism, purviews=False, cause_purviews=False,
            effect_purviews=False):
    """Return the concept specified by a mechanism within a subsytem.

    Args:
        subsystem (Subsystem): The context in which the mechanism should be
            considered.
        mechanism (tuple[int]): The candidate set of nodes.

    Keyword Args:
        purviews (tuple[tuple[int]]): Restrict the possible purviews to those
            in this list.
        cause_purviews (tuple[tuple[int]]): Restrict the possible cause
            purviews to those in this list. Takes precedence over ``purviews``.
        effect_purviews (tuple[tuple[int]]): Restrict the possible effect
            purviews to those in this list. Takes precedence over ``purviews``.

    Returns:
        Concept: The pair of maximally irreducible cause/effect repertoires
        that constitute the concept specified by the given mechanism.
    """
    start = time()
    log.debug('Computing concept %s...', mechanism)

    # If the mechanism is empty, there is no concept.
    if not mechanism:
        result = subsystem.null_concept
    else:
        result = subsystem.concept(
            mechanism, purviews=purviews, cause_purviews=cause_purviews,
            effect_purviews=effect_purviews)

    result.time = round(time() - start, config.PRECISION)
    log.debug('Found concept %s', mechanism)
    return result


class ComputeCauseEffectStructure(parallel.MapReduce):
    """Engine for computing a |CauseEffectStructure|."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Computing concepts'

    def empty_result(self, *args):
        return []

    @staticmethod
    def compute(mechanism, subsystem, purviews, cause_purviews,
                effect_purviews):
        """Compute a |Concept| for a mechanism, in this |Subsystem| with the
        provided purviews."""
        return concept(subsystem, mechanism, purviews=purviews,
                       cause_purviews=cause_purviews,
                       effect_purviews=effect_purviews)

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero |small_phi| to the
        |CauseEffectStructure|."""
        if new_concept.phi > 0:
            concepts.append(new_concept)
        return concepts


def ces(subsystem, mechanisms=False, purviews=False, cause_purviews=False,
        effect_purviews=False, parallel=False):
    """Return the conceptual structure of this subsystem, optionally restricted
    to concepts with the mechanisms and purviews given in keyword arguments.

    If you don't need the full |CauseEffectStructure|, restricting the possible
    mechanisms and purviews can make this function much faster.

    Args:
        subsystem (Subsystem): The subsystem for which to determine the
            |CauseEffectStructure|.

    Keyword Args:
        mechanisms (tuple[tuple[int]]): Restrict possible mechanisms to those
            in this list.
        purviews (tuple[tuple[int]]): Same as in :func:`concept`.
        cause_purviews (tuple[tuple[int]]): Same as in :func:`concept`.
        effect_purviews (tuple[tuple[int]]): Same as in :func:`concept`.
        parallel (bool): Whether to compute concepts in parallel. If ``True``,
            overrides :data:`config.PARALLEL_CONCEPT_EVALUATION`.

    Returns:
        CauseEffectStructure: A tuple of every |Concept| in the cause-effect
        structure.
    """
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices, nonempty=True)

    engine = ComputeCauseEffectStructure(mechanisms, subsystem, purviews,
                                         cause_purviews, effect_purviews)

    return models.CauseEffectStructure(
        engine.run(parallel or config.PARALLEL_CONCEPT_EVALUATION))


def conceptual_information(subsystem):
    """Return the conceptual information for a |Subsystem|.

    This is the distance from the subsystem's |CauseEffectStructure| to the
    null concept.
    """
    ci = ces_distance(ces(subsystem), ())
    return round(ci, config.PRECISION)
