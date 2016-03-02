#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/big_phi.py

"""
Methods for computing concepts, constellations, and integrated information of
subsystems.
"""

import functools
import logging
import multiprocessing
from time import time

import numpy as np

from . import parallel
from .concept import constellation
from .distance import constellation_distance
from .. import config, memory, utils, validate
from ..models import BigMip, Cut, _null_bigmip, _single_node_bigmip
from ..network import Network
from ..subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)


# Expose `compute.evaluate_cut` to public API
def evaluate_cut(uncut_subsystem, cut, unpartitioned_constellation):
    """Find the |BigMip| for a given cut.

    Args:
        uncut_subsystem (Subsystem): The subsystem without the cut applied.
        cut (Cut): The cut to evaluate.
        unpartitioned_constellation (Constellation): The constellation of the
            uncut subsystem.

    Returns:
        |BigMip|: The |BigMip| for that cut.
    """
    log.debug("Evaluating cut {}...".format(cut))

    cut_subsystem = Subsystem(uncut_subsystem.network,
                              uncut_subsystem.state,
                              uncut_subsystem.node_indices,
                              cut=cut,
                              mice_cache=uncut_subsystem._mice_cache)
    mechanisms = {c.mechanism for c in unpartitioned_constellation}
    if not config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms |= set(cut.all_cut_mechanisms(uncut_subsystem.node_indices))
    partitioned_constellation = constellation(cut_subsystem, mechanisms)

    log.debug("Finished evaluating cut {}.".format(cut))

    phi = constellation_distance(unpartitioned_constellation,
                                 partitioned_constellation)

    return BigMip(
        phi=round(phi, config.PRECISION),
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=partitioned_constellation,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


# Wrapper for `evaluate_cut` for parallel processing.
def _eval_wrapper(in_queue, out_queue, subsystem, unpartitioned_constellation):
    while True:
        cut = in_queue.get()
        if cut is None:
            break
        new_mip = evaluate_cut(subsystem, cut, unpartitioned_constellation)
        out_queue.put(new_mip)
    out_queue.put(None)


def _find_mip_parallel(subsystem, cuts, unpartitioned_constellation, min_mip):
    """Find the MIP for a subsystem with a parallel loop over all cuts.

    Uses the specified number of cores.
    """
    number_of_processes = parallel.get_num_processes()
    # Define input and output queues to allow short-circuit if a cut if found
    # with zero Phi. Load the input queue with all possible cuts and a 'poison
    # pill' for each process.
    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    for cut in cuts:
        in_queue.put(cut)
    for i in range(number_of_processes):
        in_queue.put(None)
    # Initialize the processes and start them.
    processes = [
        multiprocessing.Process(target=_eval_wrapper,
                                args=(in_queue, out_queue, subsystem,
                                      unpartitioned_constellation))
        for i in range(number_of_processes)
    ]
    for i in range(number_of_processes):
        processes[i].start()
    # Continue to process output queue until all processes have completed, or a
    # 'poison pill' has been returned.
    while True:
        new_mip = out_queue.get()
        if new_mip is None:
            number_of_processes -= 1
            if number_of_processes == 0:
                break
        elif new_mip.phi == 0:
            min_mip = new_mip
            for process in processes:
                process.terminate()
            break
        elif new_mip < min_mip:
            min_mip = new_mip
    return min_mip


def _find_mip_sequential(subsystem, cuts, unpartitioned_constellation,
                         min_mip):
    """Find the minimal cut for a subsystem by sequentially loop over all cuts.

    Holds only two |BigMip|s in memory at once.
    """
    for i, cut in enumerate(cuts):
        new_mip = evaluate_cut(subsystem, cut, unpartitioned_constellation)
        log.debug("Finished {} of {} cuts.".format(i + 1, len(cuts)))
        if new_mip < min_mip:
            min_mip = new_mip
        # Short-circuit as soon as we find a MIP with effectively 0 phi.
        if min_mip.phi == 0:
            break
    return min_mip


def big_mip_bipartitions(nodes):
    """Return all |big_phi| cuts for the given nodes.

    This value changes based on `config.CUT_ONE_APPROXIMATION`.

    Args:
        nodes tuple(int): The node indices to partition.
    Returns:
        list(|Cut|): All unidirectional partitions.
    """
    if config.CUT_ONE_APPROXIMATION:
        bipartitions = utils.directed_bipartition_of_one(nodes)
    else:
        # Skip the first and last (trivial, null cut) bipartitions
        bipartitions = utils.directed_bipartition(nodes)[1:-1]

    return [Cut(bipartition[0], bipartition[1])
            for bipartition in bipartitions]


# TODO document big_mip
@memory.cache(ignore=["subsystem"])
def _big_mip(cache_key, subsystem):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        |BigMip|: A nested structure containing all the data from the
        intermediate calculations. The top level contains the basic MIP
        information for the given subsystem.
    """
    log.info("Calculating big-phi data for {}...".format(subsystem))
    start = time()

    if config.PARALLEL_CUT_EVALUATION:
        _find_mip = _find_mip_parallel
    else:
        _find_mip = _find_mip_sequential

    # Annote a BigMip with the total elapsed calculation time, and optionally
    # also with the time taken to calculate the unpartitioned constellation.
    def time_annotated(big_mip, small_phi_time=0.0):
        big_mip.time = time() - start
        big_mip.small_phi_time = small_phi_time
        return big_mip

    # Special case for single-node subsystems.
    if len(subsystem) == 1:
        log.info('Single-node {}; returning the hard-coded single-node MIP '
                 'immediately.'.format(subsystem))
        return time_annotated(_single_node_bigmip(subsystem))

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty; or
    #   - an elementary mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null MIP.
    if not subsystem:
        log.info('Subsystem {} is empty; returning null MIP '
                 'immediately.'.format(subsystem))
        return time_annotated(_null_bigmip(subsystem))

    if not utils.strongly_connected(subsystem.connectivity_matrix,
                                    subsystem.node_indices):
        log.info('{} is not strongly connected; returning null MIP '
                 'immediately.'.format(subsystem))
        return time_annotated(_null_bigmip(subsystem))
    # =========================================================================

    log.debug("Finding unpartitioned constellation...")
    small_phi_start = time()
    unpartitioned_constellation = constellation(subsystem)
    small_phi_time = time() - small_phi_start
    log.debug("Found unpartitioned constellation.")

    if not unpartitioned_constellation:
        # Short-circuit if there are no concepts in the unpartitioned
        # constellation.
        result = time_annotated(_null_bigmip(subsystem))
    else:
        cuts = big_mip_bipartitions(subsystem.node_indices)
        min_mip = _null_bigmip(subsystem)
        min_mip.phi = float('inf')
        min_mip = _find_mip(subsystem, cuts, unpartitioned_constellation,
                            min_mip)
        result = time_annotated(min_mip, small_phi_time)

    log.info("Finished calculating big-phi data for {}.".format(subsystem))
    log.debug("RESULT: \n" + str(result))
    return result


# Wrapper to ensure that the cache key is the native hash of the subsystem, so
# joblib doesn't mistakenly recompute things when the subsystem's MICE cache is
# changed.
@functools.wraps(_big_mip)
def big_mip(subsystem):
    return _big_mip(hash(subsystem), subsystem)


def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return big_mip(subsystem).phi


def subsystems(network, state):
    """Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible state.
    """
    for subset in utils.powerset(network.node_indices):
        try:
            yield Subsystem(network, state, subset)
        except validate.StateUnreachableError:
            pass


def all_complexes(network, state):
    """Return a generator for all complexes of the network.

    Includes reducible, zero-phi complexes (which are not, strictly speaking,
    complexes at all).
    """
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return (big_mip(subsystem) for subsystem in subsystems(network, state))


def possible_complexes(network, state):
    """Return a generator of subsystems of a network that could be a complex.

    This is the just powerset of the nodes that have at least one input and
    output (nodes with no inputs or no outputs cannot be part of a main
    complex, because they do not have a causal link with the rest of the
    subsystem in the past or future, respectively).

    Does not include subsystems in an impossible state.

    Args:
        network (Network): The network for which to return possible complexes.
        state (tuple(int)): The state of the network.

    Yields:
        (Subsystem): The next subsystem which could be a complex.
    """
    inputs = np.sum(network.connectivity_matrix, 0)
    outputs = np.sum(network.connectivity_matrix, 1)
    nodes_have_inputs_and_outputs = np.logical_and(inputs > 0, outputs > 0)
    causally_significant_nodes = np.where(nodes_have_inputs_and_outputs)[0]

    for subset in utils.powerset(causally_significant_nodes):
        # Don't return empty system
        if len(subset) == 0:
            continue

        # Don't return subsystems that are in an impossible state.
        try:
            yield Subsystem(network, state, subset)
        except validate.StateUnreachableError:
            continue


def complexes(network, state):
    """Return a generator for all irreducible complexes of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return tuple(filter(None, (big_mip(subsystem) for subsystem in
                               possible_complexes(network, state))))


def main_complex(network, state):
    """Return the main complex of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    log.info("Calculating main complex...")
    result = complexes(network, state)
    if result:
        result = max(result)
    else:
        empty_subsystem = Subsystem(network, state, ())
        result = _null_bigmip(empty_subsystem)
    log.info("Finished calculating main complex.")
    log.debug("RESULT: \n" + str(result))
    return result


def condensed(network, state):
    """Return the set of maximal non-overlapping complexes."""
    condensed = []
    covered_nodes = set()
    for c in reversed(sorted(complexes(network, state))):
        if not any(n in covered_nodes for n in c.subsystem.node_indices):
            condensed.append(c)
            covered_nodes = covered_nodes | set(c.subsystem.node_indices)
    return condensed
