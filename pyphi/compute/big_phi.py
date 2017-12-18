# -*- coding: utf-8 -*-
# compute/big_phi.py

"""
Functions for computing integrated information and finding complexes.
"""

import functools
import logging
from time import time

from .. import (Direction, config, connectivity, exceptions, memory, utils,
                validate)
from ..models import (Concept, Cut, KCut, SystemIrreducibilityAnalysis,
                      _null_sia, cmp, fmt)
from ..partition import directed_bipartition, directed_bipartition_of_one
from ..subsystem import Subsystem, mip_partitions
from .concept import ces
from .distance import ces_distance
from .parallel import MapReduce

# Create a logger for this module.
log = logging.getLogger(__name__)


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

    phi = ces_distance(unpartitioned_ces,
                       partitioned_ces)

    return SystemIrreducibilityAnalysis(
        phi=phi,
        unpartitioned_ces=unpartitioned_ces,
        partitioned_ces=partitioned_ces,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


class ComputeSystemIrreducibility(MapReduce):
    """Computation engine for system-level irreducibility."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Evaluating {} cuts'.format(fmt.BIG_PHI)

    def empty_result(self, subsystem, unpartitioned_ces):
        """Begin with a mip with infinite |big_phi|; all actual mips will have
        less."""
        return _null_sia(subsystem, phi=float('inf'))

    @staticmethod
    def compute(cut, subsystem, unpartitioned_ces):
        """Evaluate a cut."""
        return evaluate_cut(subsystem, cut, unpartitioned_ces)

    def process_result(self, new_mip, min_mip):
        """Check if the new mip has smaller phi than the standing result."""
        if new_mip.phi == 0:
            self.done = True  # Short-circuit
            return new_mip

        elif new_mip < min_mip:
            return new_mip

        return min_mip


def sia_bipartitions(nodes):
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

    return [Cut(bipartition[0], bipartition[1])
            for bipartition in bipartitions]


def _unpartitioned_ces(subsystem):
    """Parallelize the unpartitioned |CauseEffectStructure| if parallelizing
    cuts, since we have free processors because we're not computing any cuts
    yet."""
    return ces(subsystem, parallel=config.PARALLEL_CUT_EVALUATION)


# pylint: disable=unused-argument
@memory.cache(ignore=["subsystem"])
def _sia(cache_key, subsystem):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        SystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given subsystem.
    """
    log.info('Calculating big-phi data for %s...', subsystem)
    start = time()

    def time_annotated(bm, small_phi_time=0.0):
        """Annote a |SystemIrreducibilityAnalysis| with the total elapsed
        calculation time.

        Optionally add the time taken to calculate the unpartitioned
        |CauseEffectStructure|.
        """
        bm.time = round(time() - start, config.PRECISION)
        bm.small_phi_time = round(small_phi_time, config.PRECISION)
        return bm

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
        return time_annotated(_null_sia(subsystem))

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        log.info('%s is not strongly connected; returning null SIA '
                 'immediately.', subsystem)
        return time_annotated(_null_sia(subsystem))

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            log.info('Single micro nodes %s without selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return time_annotated(_null_sia(subsystem))
        # Even if the node has a self-loop, we may still define phi to be zero.
        elif not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            log.info('Single micro nodes %s with selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return time_annotated(_null_sia(subsystem))
    # =========================================================================

    log.debug('Finding unpartitioned CauseEffectStructure...')
    small_phi_start = time()
    unpartitioned_ces = _unpartitioned_ces(subsystem)
    small_phi_time = round(time() - small_phi_start, config.PRECISION)

    if not unpartitioned_ces:
        log.info('Empty unpartitioned CauseEffectStructure; returning null '
                 'SIA immediately.')
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return time_annotated(_null_sia(subsystem))

    log.debug('Found unpartitioned CauseEffectStructure.')
    if len(subsystem.cut_indices) == 1:
        cuts = [Cut(subsystem.cut_indices, subsystem.cut_indices)]
    else:
        cuts = sia_bipartitions(subsystem.cut_indices)
    engine = ComputeSystemIrreducibility(
        cuts, subsystem, unpartitioned_ces)
    min_mip = engine.run(config.PARALLEL_CUT_EVALUATION)
    result = time_annotated(min_mip, small_phi_time)

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


def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return sia(subsystem).phi


def _reachable_subsystems(network, indices, state):
    """A generator over all subsystems in a valid state."""
    validate.is_network(network)

    # Return subsystems largest to smallest to optimize parallel
    # resource usage.
    for subset in utils.powerset(indices, nonempty=True, reverse=True):
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            pass


def subsystems(network, state):
    """Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible state.
    """
    return _reachable_subsystems(network, network.node_indices, state)


def possible_complexes(network, state):
    """Return a generator of subsystems of a network that could be a complex.

    This is the just powerset of the nodes that have at least one input and
    output (nodes with no inputs or no outputs cannot be part of a main
    complex, because they do not have a causal link with the rest of the
    subsystem in the previous or next timestep, respectively).

    Does not include subsystems in an impossible state.

    Args:
        network (Network): The network for which to return possible complexes.
        state (tuple[int]): The state of the network.

    Yields:
        Subsystem: The next subsystem which could be a complex.
    """
    return _reachable_subsystems(network, network.causally_significant_nodes,
                                 state)

class FindAllComplexes(MapReduce):
    """Computation engine for finding all complexes."""
    # pylint: disable=unused-argument,arguments-differ
    description = 'Finding complexes'

    def empty_result(self):
        return []

    @staticmethod
    def compute(subsystem):
        return sia(subsystem)

    def process_result(self, new_sia, sias):
        sias.append(new_sia)
        return sias


def all_complexes(network, state):
    """Return a generator for all complexes of the network.

    Includes reducible, zero-|big_phi| complexes (which are not, strictly
    speaking, complexes at all).
    """
    engine = FindAllComplexes(subsystems(network, state))
    return engine.run(config.PARALLEL_COMPLEX_EVALUATION)


class FindIrreducibleComplexes(FindAllComplexes):
    """Computation engine for finding irreducible complexes of a network."""

    def process_result(self, new_sia, sias):
        if new_sia.phi > 0:
            sias.append(new_sia)
        return sias


def complexes(network, state):
    """Return all irreducible complexes of the network."""
    engine = FindIrreducibleComplexes(possible_complexes(network, state))
    return engine.run(config.PARALLEL_COMPLEX_EVALUATION)


def major_complex(network, state):
    """Return the major complex of the network."""
    log.info('Calculating major complex...')

    result = complexes(network, state)
    if result:
        result = max(result)
    else:
        empty_subsystem = Subsystem(network, state, ())
        result = _null_sia(empty_subsystem)

    log.info("Finished calculating major complex.")

    return result


def condensed(network, state):
    """Return the set of maximal non-overlapping complexes."""
    result = []
    covered_nodes = set()

    for c in reversed(sorted(complexes(network, state))):
        if not any(n in covered_nodes for n in c.subsystem.node_indices):
            result.append(c)
            covered_nodes = covered_nodes | set(c.subsystem.node_indices)

    return result


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
        """Compute a concept, using the appropriate system for each side of
        the cut."""
        cause = self.cause_system.mic(
            mechanism, purviews=(cause_purviews or purviews))

        effect = self.effect_system.mie(
            mechanism, purviews=(effect_purviews or purviews))

        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self)

    def __str__(self):
        return 'ConceptStyleSystem{}'.format(self.node_indices)


def concept_cuts(direction, node_indices):
    """Generator over all concept-syle cuts for these nodes."""
    for partition in mip_partitions(node_indices, node_indices):
        yield KCut(direction, partition)


def directional_sia(subsystem, direction, unpartitioned_ces=None):
    """Calculate a concept-style SystemIrreducibilityAnalysisCause or
    SystemIrreducibilityAnalysisEffect."""
    if unpartitioned_ces is None:
        unpartitioned_ces = _unpartitioned_ces(subsystem)

    c_system = ConceptStyleSystem(subsystem, direction)
    cuts = concept_cuts(direction, c_system.cut_indices)

    # Run the default SIA engine
    # TODO: verify that short-cutting works correctly?
    engine = ComputeSystemIrreducibility(
        cuts, c_system, unpartitioned_ces)
    return engine.run(config.PARALLEL_CUT_EVALUATION)


# TODO: only return the minimal mip, instead of both
class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):
    """Represents a |SIA| computed using concept-style system cuts."""

    def __init__(self, mip_cause, mip_effect):
        self.sia_cause = mip_cause
        self.sia_effect = mip_effect

    @property
    def min_mip(self):
        return min(self.sia_cause, self.sia_effect, key=lambda m: m.phi)

    def __getattr__(self, name):
        """Pass attribute access through to the minimal mip."""
        if ('sia_cause' in self.__dict__ and 'sia_effect' in self.__dict__):
            return getattr(self.min_mip, name)
        raise AttributeError(name)

    def __eq__(self, other):
        return cmp.general_eq(self, other, ['phi'])

    unorderable_unless_eq = ['network']

    def order_by(self):
        return [self.phi, len(self.subsystem)]

    def __repr__(self):
        return repr(self.min_mip)

    def __str__(self):
        return str(self.min_mip)


# TODO: cache
def sia_concept_style(subsystem):
    """Compute a concept-style SystemIrreducibilityAnalysis"""
    unpartitioned_ces = _unpartitioned_ces(subsystem)

    mip_cause = directional_sia(subsystem, Direction.CAUSE,
                                unpartitioned_ces)
    mip_effect = directional_sia(subsystem, Direction.EFFECT,
                                 unpartitioned_ces)

    return SystemIrreducibilityAnalysisConceptStyle(mip_cause, mip_effect)
