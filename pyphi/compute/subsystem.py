#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/subsystem.py

"""
Functions for computing subsystem-level properties.
"""

import functools
import logging

from .. import Direction, config, connectivity, memory, utils
from ..models import (CauseEffectStructure, Concept, Cut, KCut,
                      SystemIrreducibilityAnalysis, _null_sia, cmp, fmt)
from ..partition import (directed_bipartition, directed_bipartition_of_one,
                         mip_partitions)
from ..utils import time_annotated
from .distance import ces_distance
from .parallel import MapReduce

# Create a logger for this module.
log = logging.getLogger(__name__)


class ComputeCauseEffectStructure(MapReduce):
    """Engine for computing a |CauseEffectStructure|."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Computing concepts'

    @property
    def subsystem(self):
        return self.context[0]

    def empty_result(self, *args):
        return []

    @staticmethod
    def compute(mechanism, subsystem, purviews, cause_purviews,
                effect_purviews):
        """Compute a |Concept| for a mechanism, in this |Subsystem| with the
        provided purviews.
        """
        concept = subsystem.concept(mechanism,
                                    purviews=purviews,
                                    cause_purviews=cause_purviews,
                                    effect_purviews=effect_purviews)
        # Don't serialize the subsystem.
        # This is replaced on the other side of the queue, and ensures
        # that all concepts in the CES reference the same subsystem.
        concept.subsystem = None
        return concept

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero |small_phi| to the
        |CauseEffectStructure|.
        """
        if new_concept.phi > 0:
            # Replace the subsystem
            new_concept.subsystem = self.subsystem
            concepts.append(new_concept)
        return concepts


@time_annotated
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
        purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        cause_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        effect_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
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

    return CauseEffectStructure(engine.run(parallel or
                                           config.PARALLEL_CONCEPT_EVALUATION),
                                subsystem=subsystem)


def conceptual_info(subsystem):
    """Return the conceptual information for a |Subsystem|.

    This is the distance from the subsystem's |CauseEffectStructure| to the
    null concept.
    """
    ci = ces_distance(ces(subsystem),
                      CauseEffectStructure((), subsystem=subsystem))
    return round(ci, config.PRECISION)


def evaluate_cut(uncut_subsystem, cut, unpartitioned_ces):
    """Compute the system irreducibility for a given cut.

    Args:
        uncut_subsystem (Subsystem): The subsystem without the cut applied.
        cut (Cut): The cut to evaluate.
        unpartitioned_ces (CauseEffectStructure): The cause-effect structure of
            the uncut subsystem.

    Returns:
        SystemIrreducibilityAnalysis: The |SystemIrreducibilityAnalysis| for
        that cut.
    """
    log.debug('Evaluating %s...', cut)

    cut_subsystem = uncut_subsystem.apply_cut(cut)

    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = unpartitioned_ces.mechanisms
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the cut divides the mechanism.
        mechanisms = set(
            unpartitioned_ces.mechanisms +
            list(cut_subsystem.cut_mechanisms))

    partitioned_ces = ces(cut_subsystem, mechanisms)

    log.debug('Finished evaluating %s.', cut)

    phi_ = ces_distance(unpartitioned_ces, partitioned_ces)

    return SystemIrreducibilityAnalysis(
        phi=phi_,
        ces=unpartitioned_ces,
        partitioned_ces=partitioned_ces,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


class ComputeSystemIrreducibility(MapReduce):
    """Computation engine for system-level irreducibility."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Evaluating {} cuts'.format(fmt.BIG_PHI)

    def empty_result(self, subsystem, unpartitioned_ces):
        """Begin with a |SIA| with infinite |big_phi|; all actual SIAs will
        have less.
        """
        return _null_sia(subsystem, phi=float('inf'))

    @staticmethod
    def compute(cut, subsystem, unpartitioned_ces):
        """Evaluate a cut."""
        return evaluate_cut(subsystem, cut, unpartitioned_ces)

    def process_result(self, new_sia, min_sia):
        """Check if the new SIA has smaller |big_phi| than the standing
        result.
        """
        if new_sia.phi == 0:
            self.done = True  # Short-circuit
            return new_sia

        elif new_sia < min_sia:
            return new_sia

        return min_sia


def sia_bipartitions(nodes, node_labels=None):
    """Return all |big_phi| cuts for the given nodes.

    This value changes based on :const:`config.CUT_ONE_APPROXIMATION`.

    Args:
        nodes (tuple[int]): The node indices to partition.
    Returns:
        list[Cut]: All unidirectional partitions.
    """
    if config.CUT_ONE_APPROXIMATION:
        bipartitions = directed_bipartition_of_one(nodes)
    else:
        # Don't consider trivial partitions where one part is empty
        bipartitions = directed_bipartition(nodes, nontrivial=True)

    return [Cut(bipartition[0], bipartition[1], node_labels)
            for bipartition in bipartitions]


def _ces(subsystem):
    """Parallelize the unpartitioned |CauseEffectStructure| if parallelizing
    cuts, since we have free processors because we're not computing any cuts
    yet.
    """
    return ces(subsystem, parallel=config.PARALLEL_CUT_EVALUATION)


@memory.cache(ignore=["subsystem"])
@time_annotated
def _sia(cache_key, subsystem):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        SystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given subsystem.
    """
    # pylint: disable=unused-argument

    log.info('Calculating big-phi data for %s...', subsystem)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    if not subsystem:
        log.info('Subsystem %s is empty; returning null SIA '
                 'immediately.', subsystem)
        return _null_sia(subsystem)

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        log.info('%s is not strongly connected; returning null SIA '
                 'immediately.', subsystem)
        return _null_sia(subsystem)

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            log.info('Single micro nodes %s without selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return _null_sia(subsystem)
        # Even if the node has a self-loop, we may still define phi to be zero.
        elif not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            log.info('Single micro nodes %s with selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return _null_sia(subsystem)
    # =========================================================================

    log.debug('Finding unpartitioned CauseEffectStructure...')
    unpartitioned_ces = _ces(subsystem)

    if not unpartitioned_ces:
        log.info('Empty unpartitioned CauseEffectStructure; returning null '
                 'SIA immediately.')
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return _null_sia(subsystem)

    log.debug('Found unpartitioned CauseEffectStructure.')

    # TODO: move this into sia_bipartitions?
    # Only True if SINGLE_MICRO_NODES...=True, no?
    if len(subsystem.cut_indices) == 1:
        cuts = [Cut(subsystem.cut_indices, subsystem.cut_indices,
                    subsystem.cut_node_labels)]
    else:
        cuts = sia_bipartitions(subsystem.cut_indices,
                                subsystem.cut_node_labels)

    engine = ComputeSystemIrreducibility(
        cuts, subsystem, unpartitioned_ces)
    result = engine.run(config.PARALLEL_CUT_EVALUATION)

    if config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA:
        log.debug('Clearing subsystem caches.')
        subsystem.clear_caches()

    log.info('Finished calculating big-phi data for %s.', subsystem)

    return result


# TODO(maintainance): don't forget to add any new configuration options here if
# they can change big-phi values
def _sia_cache_key(subsystem):
    """The cache key of the subsystem.

    This includes the native hash of the subsystem and all configuration values
    which change the results of ``sia``.
    """
    return (
        hash(subsystem),
        config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS,
        config.CUT_ONE_APPROXIMATION,
        config.MEASURE,
        config.PRECISION,
        config.VALIDATE_SUBSYSTEM_STATES,
        config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI,
        config.PARTITION_TYPE,
    )


# Wrapper to ensure that the cache key is the native hash of the subsystem, so
# joblib doesn't mistakenly recompute things when the subsystem's MICE cache is
# changed. The cache is also keyed on configuration values which affect the
# value of the computation.
@functools.wraps(_sia)
def sia(subsystem):  # pylint: disable=missing-docstring
    if config.SYSTEM_CUTS == 'CONCEPT_STYLE':
        return sia_concept_style(subsystem)

    return _sia(_sia_cache_key(subsystem), subsystem)


def phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return sia(subsystem).phi


class ConceptStyleSystem:
    """A functional replacement for ``Subsystem`` implementing concept-style
    system cuts.
    """

    def __init__(self, subsystem, direction, cut=None):
        self.subsystem = subsystem
        self.direction = direction
        self.cut = cut
        self.cut_system = subsystem.apply_cut(cut)

    def apply_cut(self, cut):
        return ConceptStyleSystem(self.subsystem, self.direction, cut)

    def __getattr__(self, name):
        """Pass attribute access through to the basic subsystem."""
        # Unpickling calls `__getattr__` before the object's dict is populated;
        # check that `subsystem` exists to avoid a recursion error.
        # See https://bugs.python.org/issue5370.
        if 'subsystem' in self.__dict__:
            return getattr(self.subsystem, name)
        raise AttributeError(name)

    def __len__(self):
        return len(self.subsystem)

    @property
    def cause_system(self):
        return {
            Direction.CAUSE: self.cut_system,
            Direction.EFFECT: self.subsystem
        }[self.direction]

    @property
    def effect_system(self):
        return {
            Direction.CAUSE: self.subsystem,
            Direction.EFFECT: self.cut_system
        }[self.direction]

    def concept(self, mechanism, purviews=False, cause_purviews=False,
                effect_purviews=False):
        """Compute a concept, using the appropriate system for each side of the
        cut.
        """
        cause = self.cause_system.mic(
            mechanism, purviews=(cause_purviews or purviews))

        effect = self.effect_system.mie(
            mechanism, purviews=(effect_purviews or purviews))

        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self)

    def __str__(self):
        return 'ConceptStyleSystem{}'.format(self.node_indices)


def concept_cuts(direction, node_indices, node_labels=None):
    """Generator over all concept-syle cuts for these nodes."""
    for partition in mip_partitions(node_indices, node_indices):
        yield KCut(direction, partition, node_labels)


def directional_sia(subsystem, direction, unpartitioned_ces=None):
    """Calculate a concept-style SystemIrreducibilityAnalysisCause or
    SystemIrreducibilityAnalysisEffect.
    """
    if unpartitioned_ces is None:
        unpartitioned_ces = _ces(subsystem)

    c_system = ConceptStyleSystem(subsystem, direction)
    cuts = concept_cuts(direction, c_system.cut_indices, subsystem.node_labels)

    # Run the default SIA engine
    # TODO: verify that short-cutting works correctly?
    engine = ComputeSystemIrreducibility(
        cuts, c_system, unpartitioned_ces)
    return engine.run(config.PARALLEL_CUT_EVALUATION)


# TODO: only return the minimal SIA, instead of both
class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):
    """Represents a |SIA| computed using concept-style system cuts."""

    def __init__(self, sia_cause, sia_effect):
        self.sia_cause = sia_cause
        self.sia_effect = sia_effect

    @property
    def min_sia(self):
        return min(self.sia_cause, self.sia_effect, key=lambda m: m.phi)

    def __getattr__(self, name):
        """Pass attribute access through to the minimal SIA."""
        if ('sia_cause' in self.__dict__ and 'sia_effect' in self.__dict__):
            return getattr(self.min_sia, name)
        raise AttributeError(name)

    def __eq__(self, other):
        return cmp.general_eq(self, other, ['phi'])

    unorderable_unless_eq = ['network']

    def order_by(self):
        return [self.phi, len(self.subsystem)]

    def __repr__(self):
        return repr(self.min_sia)

    def __str__(self):
        return str(self.min_sia)


# TODO: cache
def sia_concept_style(subsystem):
    """Compute a concept-style SystemIrreducibilityAnalysis"""
    unpartitioned_ces = _ces(subsystem)

    sia_cause = directional_sia(subsystem, Direction.CAUSE,
                                unpartitioned_ces)
    sia_effect = directional_sia(subsystem, Direction.EFFECT,
                                 unpartitioned_ces)

    return SystemIrreducibilityAnalysisConceptStyle(sia_cause, sia_effect)
