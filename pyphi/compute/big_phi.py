# -*- coding: utf-8 -*-
# compute/big_phi.py

'''
Functions for computing integrated information and finding complexes.
'''

import functools
import logging
from time import time

from .. import (Direction, config, connectivity, exceptions, memory, utils,
                validate)
from ..models import BigMip, Concept, Cut, KCut, _null_bigmip, cmp, fmt
from ..partition import directed_bipartition, directed_bipartition_of_one
from ..subsystem import Subsystem, mip_partitions
from .concept import constellation
from .distance import constellation_distance
from .parallel import MapReduce

# Create a logger for this module.
log = logging.getLogger(__name__)


def evaluate_cut(uncut_subsystem, cut, unpartitioned_constellation):
    '''Find the |BigMip| for a given cut.

    Args:
        uncut_subsystem (Subsystem): The subsystem without the cut applied.
        cut (Cut): The cut to evaluate.
        unpartitioned_constellation (Constellation): The constellation of the
            uncut subsystem.

    Returns:
        BigMip: The |BigMip| for that cut.
    '''
    log.debug('Evaluating %s...', cut)

    cut_subsystem = uncut_subsystem.apply_cut(cut)

    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = unpartitioned_constellation.mechanisms
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the cut divides the mechanism.
        mechanisms = set(
            unpartitioned_constellation.mechanisms +
            list(cut_subsystem.cut_mechanisms))

    partitioned_constellation = constellation(cut_subsystem, mechanisms)

    log.debug('Finished evaluating %s.', cut)

    phi = constellation_distance(unpartitioned_constellation,
                                 partitioned_constellation)

    return BigMip(
        phi=phi,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=partitioned_constellation,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


class FindMip(MapReduce):
    '''Computation engine for finding the minimal |BigMip|.'''
    # pylint: disable=unused-argument,arguments-differ

    description = 'Evaluating {} cuts'.format(fmt.BIG_PHI)

    def empty_result(self, subsystem, unpartitioned_constellation):
        '''Begin with a mip with infinite |big_phi|; all actual mips will have
        less.'''
        return _null_bigmip(subsystem, phi=float('inf'))

    @staticmethod
    def compute(cut, subsystem, unpartitioned_constellation):
        '''Evaluate a cut.'''
        return evaluate_cut(subsystem, cut, unpartitioned_constellation)

    def process_result(self, new_mip, min_mip):
        '''Check if the new mip has smaller phi than the standing result.'''
        if new_mip.phi == 0:
            self.done = True  # Short-circuit
            return new_mip

        elif new_mip < min_mip:
            return new_mip

        return min_mip


def big_mip_bipartitions(nodes):
    '''Return all |big_phi| cuts for the given nodes.

    This value changes based on :const:`config.CUT_ONE_APPROXIMATION`.

    Args:
        nodes (tuple[int]): The node indices to partition.
    Returns:
        list[Cut]: All unidirectional partitions.
    '''
    if config.CUT_ONE_APPROXIMATION:
        bipartitions = directed_bipartition_of_one(nodes)
    else:
        # Don't consider trivial partitions where one part is empty
        bipartitions = directed_bipartition(nodes, nontrivial=True)

    return [Cut(bipartition[0], bipartition[1])
            for bipartition in bipartitions]


def _unpartitioned_constellation(subsystem):
    '''Parallelize the unpartitioned constellation if parallelizing cuts,
    since we have free processors because we're not computing any cuts yet.'''
    return constellation(subsystem, parallel=config.PARALLEL_CUT_EVALUATION)


# pylint: disable=unused-argument
@memory.cache(ignore=["subsystem"])
def _big_mip(cache_key, subsystem):
    '''Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        BigMip: A nested structure containing all the data from the
        intermediate calculations. The top level contains the basic MIP
        information for the given subsystem.
    '''
    log.info('Calculating big-phi data for %s...', subsystem)
    start = time()

    def time_annotated(bm, small_phi_time=0.0):
        '''Annote a BigMip with the total elapsed calculation time.

        Optionally add the time taken to calculate the unpartitioned
        constellation.
        '''
        bm.time = round(time() - start, config.PRECISION)
        bm.small_phi_time = round(small_phi_time, config.PRECISION)
        return bm

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null MIP.
    if not subsystem:
        log.info('Subsystem %s is empty; returning null MIP '
                 'immediately.', subsystem)
        return time_annotated(_null_bigmip(subsystem))

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        log.info('%s is not strongly connected; returning null MIP '
                 'immediately.', subsystem)
        return time_annotated(_null_bigmip(subsystem))

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            log.info('Single micro nodes %s without selfloops cannot have phi; '
                     'returning null MIP immediately.', subsystem)
            return time_annotated(_null_bigmip(subsystem))
        # Even if the node has a self-loop, we may still define phi to be zero.
        elif not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            log.info('Single micro nodes %s with selfloops cannot have phi; '
                     'returning null MIP immediately.', subsystem)
            return time_annotated(_null_bigmip(subsystem))
    # =========================================================================

    log.debug('Finding unpartitioned constellation...')
    small_phi_start = time()
    unpartitioned_constellation = _unpartitioned_constellation(subsystem)
    small_phi_time = round(time() - small_phi_start, config.PRECISION)

    if not unpartitioned_constellation:
        log.info('Empty unpartitioned constellation; returning null MIP '
                 'immediately.')
        # Short-circuit if there are no concepts in the unpartitioned
        # constellation.
        return time_annotated(_null_bigmip(subsystem))

    log.debug('Found unpartitioned constellation.')
    if len(subsystem.cut_indices) == 1:
        cuts = [Cut(subsystem.cut_indices, subsystem.cut_indices)]
    else:
        cuts = big_mip_bipartitions(subsystem.cut_indices)
    finder = FindMip(cuts, subsystem, unpartitioned_constellation)
    min_mip = finder.run(config.PARALLEL_CUT_EVALUATION)
    result = time_annotated(min_mip, small_phi_time)

    log.info('Finished calculating big-phi data for %s.', subsystem)

    return result


# TODO(maintainance): don't forget to add any new configuration options here if
# they can change big-phi values
def _big_mip_cache_key(subsystem):
    '''The cache key of the subsystem.

    This includes the native hash of the subsystem and all configuration values
    which change the results of ``big_mip``.
    '''
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
@functools.wraps(_big_mip)
def big_mip(subsystem):  # pylint: disable=missing-docstring
    if config.SYSTEM_CUTS == 'CONCEPT_STYLE':
        return big_mip_concept_style(subsystem)

    return _big_mip(_big_mip_cache_key(subsystem), subsystem)


def big_phi(subsystem):
    '''Return the |big_phi| value of a subsystem.'''
    return big_mip(subsystem).phi


def _reachable_subsystems(network, indices, state):
    '''A generator over all subsystems in a valid state.'''
    validate.is_network(network)

    # Return subsystems largest to smallest to optimize parallel
    # resource usage.
    for subset in utils.powerset(indices, nonempty=True, reverse=True):
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            pass


def subsystems(network, state):
    '''Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible state.
    '''
    return _reachable_subsystems(network, network.node_indices, state)


def possible_complexes(network, state):
    '''Return a generator of subsystems of a network that could be a complex.

    This is the just powerset of the nodes that have at least one input and
    output (nodes with no inputs or no outputs cannot be part of a main
    complex, because they do not have a causal link with the rest of the
    subsystem in the past or future, respectively).

    Does not include subsystems in an impossible state.

    Args:
        network (Network): The network for which to return possible complexes.
        state (tuple[int]): The state of the network.

    Yields:
        Subsystem: The next subsystem which could be a complex.
    '''
    return _reachable_subsystems(network, network.causally_significant_nodes,
                                 state)

class FindAllComplexes(MapReduce):
    '''Computation engine for computing all complexes'''
    # pylint: disable=unused-argument,arguments-differ
    description = 'Finding complexes'

    def empty_result(self):
        return []

    @staticmethod
    def compute(subsystem):
        return big_mip(subsystem)

    def process_result(self, new_big_mip, big_mips):
        big_mips.append(new_big_mip)
        return big_mips


def all_complexes(network, state):
    '''Return a generator for all complexes of the network.

    Includes reducible, zero-|big_phi| complexes (which are not, strictly
    speaking, complexes at all).
    '''
    engine = FindAllComplexes(subsystems(network, state))
    return engine.run(config.PARALLEL_COMPLEX_EVALUATION)


class FindIrreducibleComplexes(FindAllComplexes):
    '''Computation engine for computing irreducible complexes of a network.'''

    def process_result(self, new_big_mip, big_mips):
        if new_big_mip.phi > 0:
            big_mips.append(new_big_mip)
        return big_mips


def complexes(network, state):
    '''Return all irreducible complexes of the network.'''
    engine = FindIrreducibleComplexes(possible_complexes(network, state))
    return engine.run(config.PARALLEL_COMPLEX_EVALUATION)


def main_complex(network, state):
    '''Return the main complex of the network.'''
    log.info('Calculating main complex...')

    result = complexes(network, state)
    if result:
        result = max(result)
    else:
        empty_subsystem = Subsystem(network, state, ())
        result = _null_bigmip(empty_subsystem)

    log.info("Finished calculating main complex.")

    return result


def condensed(network, state):
    '''Return the set of maximal non-overlapping complexes.'''
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
            Direction.PAST: self.cut_system,
            Direction.FUTURE: self.subsystem
        }[self.direction]

    @property
    def effect_system(self):
        return {
            Direction.PAST: self.subsystem,
            Direction.FUTURE: self.cut_system
        }[self.direction]

    def concept(self, mechanism, purviews=False, past_purviews=False,
                future_purviews=False):
        '''Compute a concept, using the appropriate system for each side of
        the cut.'''
        cause = self.cause_system.core_cause(
            mechanism, purviews=(past_purviews or purviews))

        effect = self.effect_system.core_effect(
            mechanism, purviews=(future_purviews or purviews))

        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self)

    def __str__(self):
        return 'ConceptStyleSystem{}'.format(self.node_indices)


def concept_cuts(direction, node_indices):
    '''Generator over all concept-syle cuts for these nodes.'''
    for partition in mip_partitions(node_indices, node_indices):
        yield KCut(direction, partition)


def directional_big_mip(subsystem, direction, unpartitioned_constellation=None):
    """Calculate a concept-style BigMipPast or BigMipFuture."""
    if unpartitioned_constellation is None:
        unpartitioned_constellation = _unpartitioned_constellation(subsystem)

    c_system = ConceptStyleSystem(subsystem, direction)
    cuts = concept_cuts(direction, c_system.cut_indices)

    # Run the default MIP finder
    # TODO: verify that short-cutting works correctly?
    finder = FindMip(cuts, c_system, unpartitioned_constellation)
    return finder.run(config.PARALLEL_CUT_EVALUATION)


# TODO: only return the minimal mip, instead of both
class BigMipConceptStyle(cmp.Orderable):
    '''Represents a Big Mip computed using concept-style system cuts.'''

    def __init__(self, mip_past, mip_future):
        self.big_mip_past = mip_past
        self.big_mip_future = mip_future

    @property
    def min_mip(self):
        return min(self.big_mip_past, self.big_mip_future, key=lambda m: m.phi)

    def __getattr__(self, name):
        '''Pass attribute access through to the minimal mip.'''
        if ('big_mip_past' in self.__dict__ and
                'big_mip_future' in self.__dict__):
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
def big_mip_concept_style(subsystem):
    '''Compute a concept-style Big Mip'''
    unpartitioned_constellation = _unpartitioned_constellation(subsystem)

    mip_past = directional_big_mip(subsystem, Direction.PAST,
                                   unpartitioned_constellation)
    mip_future = directional_big_mip(subsystem, Direction.FUTURE,
                                     unpartitioned_constellation)

    return BigMipConceptStyle(mip_past, mip_future)
