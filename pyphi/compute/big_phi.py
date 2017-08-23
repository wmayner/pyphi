# -*- coding: utf-8 -*-
# compute/big_phi.py

'''
Functions for computing integrated information and finding complexes.
'''

import functools
import logging
from time import time

from .. import config, connectivity, exceptions, memory, utils, validate
from ..models import BigMip, Cut, _null_bigmip
from ..partition import directed_bipartition, directed_bipartition_of_one
from ..subsystem import Subsystem
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


# pylint: disable=unused-argument,arguments-differ
class FindMip(MapReduce):
    '''Computation engine for finding the minimal |BigMip|.'''
    description = 'Evaluating \u03D5 cuts'

    def empty_result(self, subsystem, unpartitioned_constellation):
        '''Begin with a mip with infinite |big_phi|; all actual mips will have
        less.'''
        return _null_bigmip(subsystem, phi=float('inf'))

    def compute(self, cut, subsystem, unpartitioned_constellation):
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
# pylint: enable=unused-argument,arguments-differ


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
    # Parallelize the unpartitioned constellation if parallelizing cuts, since
    # we have free processors because we're not computing any cuts yet.
    unpartitioned_constellation = constellation(subsystem,
                                                parallel=config.PARALLEL_CUT_EVALUATION)
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
        config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI
    )


# Wrapper to ensure that the cache key is the native hash of the subsystem, so
# joblib doesn't mistakenly recompute things when the subsystem's MICE cache is
# changed. The cache is also keyed on configuration values which affect the
# value of the computation.
@functools.wraps(_big_mip)
def big_mip(subsystem):  # pylint: disable=missing-docstring
    return _big_mip(_big_mip_cache_key(subsystem), subsystem)


def big_phi(subsystem):
    '''Return the |big_phi| value of a subsystem.'''
    return big_mip(subsystem).phi


def subsystems(network, state):
    '''Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible state.
    '''
    validate.is_network(network)

    for subset in utils.powerset(network.node_indices, nonempty=True):
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            pass


def all_complexes(network, state):
    '''Return a generator for all complexes of the network.

    Includes reducible, zero-|big_phi| complexes (which are not, strictly
    speaking, complexes at all).
    '''
    return (big_mip(subsystem) for subsystem in subsystems(network, state))


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
    validate.is_network(network)

    for subset in utils.powerset(network.causally_significant_nodes,
                                 nonempty=True):
        # Don't return subsystems that are in an impossible state.
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            continue


# pylint: disable=unused-argument,arguments-differ,redefined-outer-name
class FindComplexes(MapReduce):
    '''Computation engine for computing irreducible complexes of a network.'''
    description = 'Finding complexes'

    def empty_result(self):
        return []

    def compute(self, subsystem):
        return big_mip(subsystem)

    def process_result(self, new_big_mip, complexes):
        if new_big_mip.phi > 0:
            complexes.append(new_big_mip)
        return complexes
# pylint: enable=unused-argument,arguments-differ,redefined-outer-name


def complexes(network, state):
    '''Return all irreducible complexes of the network.'''
    engine = FindComplexes(possible_complexes(network, state))
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
