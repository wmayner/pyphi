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
import threading
from time import time

from tqdm import tqdm

from . import parallel
from .concept import constellation
from .distance import constellation_distance
from .. import config, exceptions, memory, utils, validate
from ..models import BigMip, Cut, _null_bigmip, _single_node_bigmip
from ..subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)


# Expose `compute.evaluate_cut` to public API
def evaluate_cut(uncut_subsystem, cut, unpartitioned_constellation):
    """Find the |BigMip| for a given cut.

    Args:
        uncut_subsystem (|Subsystem|): The subsystem without the cut applied.
        cut (|Cut|): The cut to evaluate.
        unpartitioned_constellation (|Constellation|): The constellation of the
            uncut subsystem.

    Returns:
        |BigMip|: The |BigMip| for that cut.
    """
    log.debug("Evaluating cut {}...".format(cut))

    cut_subsystem = uncut_subsystem.apply_cut(cut)

    from .. import macro

    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = {c.mechanism for c in unpartitioned_constellation}

    elif isinstance(uncut_subsystem, macro.MacroSubsystem):
        mechanisms = {c.mechanism for c in unpartitioned_constellation}
        for mechanism in utils.powerset(uncut_subsystem.node_indices):
            micro_mechanism = uncut_subsystem.macro2micro(mechanism)
            if cut.splits_mechanism(micro_mechanism):
                mechanisms.add(mechanism)
    else:
        mechanisms = set(
            [c.mechanism for c in unpartitioned_constellation] +
            list(cut.all_cut_mechanisms()))
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


# The worker configuration is done at the start of the worker process run.
def configure_worker(queue):
    config_worker = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'queue': {
                'class': 'logging.handlers.QueueHandler',
                'queue': queue,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['queue']
        },
    }
    logging.config.dictConfig(config_worker)


def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


POISON_PILL = None


class MapReduce:
    """
    Performs a parallel computation over an iterable.
    """
    # *args are (subsystem, unpartitioned_constellation)
    def __init__(self, iterable, default_result, *args):
        self.iterable = iterable
        self.default_result = default_result
        self.context = args

        self.number_of_processes = parallel.get_num_processes()
        # Define input and output queues to allow short-circuit if a cut if found
        # with zero Phi. Load the input queue with all possible cuts and a 'poison
        # pill' for each process.
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.log_queue = multiprocessing.Queue()

        for obj in iterable:
            self.in_queue.put(obj)

        for i in range(self.number_of_processes):
            self.in_queue.put(POISON_PILL)

        args = (self.in_queue, self.out_queue, self.log_queue) + self.context
        self.processes = [
            multiprocessing.Process(target=self.worker, args=args)
            for i in range(self.number_of_processes)]

        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue,))

        # Initialize progress bar
        self.progress = tqdm(total=len(self.iterable), leave=False,
                        disable=(not config.PROGRESS_BARS),
                        desc='Evaluating \u03D5 cuts')

    # TODO: should this not be a method? Is there a performance cost to
    # using a bound method as a Process?
    def worker(self, in_queue, out_queue, log_queue, *context):
        """Worker process."""
        configure_worker(log_queue)
        while True:
            cut = in_queue.get()
            if cut is POISON_PILL:
                break
            out_queue.put(self.compute(cut, *context))
        out_queue.put(POISON_PILL)

    def compute(self, cut, subsystem, unpartitioned_constellation):
        """Do the actual work of the parallel computation."""
        return evaluate_cut(subsystem, cut, unpartitioned_constellation)

    def process_result(self, new_mip, min_mip):
        """Process a result pulled from the queue, returning the new result."""
        if new_mip.phi == 0:
            self.finish()  # Short-circuit
            return new_mip

        elif new_mip < min_mip:
            return new_mip

        return min_mip

    def start(self):
        """Start all processses and the logger thread."""
        self.working = True

        for process in self.processes:
            process.start()

        self.log_thread.start()

    def finish(self):
        """Terminate all processes."""
        # Exit early from the execution loop
        self.working = False

        # Remove the progress bar
        self.progress.close()

        for process in self.processes:
            process.terminate()

        # Shutdown the log thread
        self.log_queue.put(POISON_PILL)
        self.log_thread.join()

    def run(self):
        """
        Performs the parallel computation, reading results from the output
        queue and passing them to ``process_result``.
        """
        self.start()

        result = self.default_result

        while self.working:
            r = self.out_queue.get()
            if r is POISON_PILL:
                self.number_of_processes -= 1
                if self.number_of_processes == 0:
                    break
            else:
                self.progress.update(1)
                result = self.process_result(r, result)

        self.finish()

        return result


def _find_mip_parallel(subsystem, cuts, unpartitioned_constellation, min_mip):

    parallelizer = MapReduce(cuts, min_mip, subsystem, unpartitioned_constellation)

    return parallelizer.run()


def _find_mip_sequential(subsystem, cuts, unpartitioned_constellation,
                         min_mip):
    """Find the minimal cut for a subsystem by sequentially loop over all cuts.

    Holds only two |BigMip|s in memory at once.
    """
    for i, cut in enumerate(progress(cuts)):
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
        nodes (tuple[int]): The node indices to partition.
    Returns:
        list[|Cut|]: All unidirectional partitions.
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
        big_mip.time = round(time() - start, config.PRECISION)
        big_mip.small_phi_time = round(small_phi_time, config.PRECISION)
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

    if not utils.strongly_connected(subsystem.cm, subsystem.node_indices):
        log.info('{} is not strongly connected; returning null MIP '
                 'immediately.'.format(subsystem))
        return time_annotated(_null_bigmip(subsystem))
    # =========================================================================

    log.debug("Finding unpartitioned constellation...")
    small_phi_start = time()
    unpartitioned_constellation = constellation(subsystem)
    small_phi_time = round(time() - small_phi_start, config.PRECISION)
    log.debug("Found unpartitioned constellation.")

    if not unpartitioned_constellation:
        # Short-circuit if there are no concepts in the unpartitioned
        # constellation.
        result = time_annotated(_null_bigmip(subsystem))
    else:
        cuts = big_mip_bipartitions(subsystem.cut_indices)
        min_mip = _null_bigmip(subsystem)
        min_mip.phi = float('inf')
        min_mip = _find_mip(subsystem, cuts, unpartitioned_constellation,
                            min_mip)
        result = time_annotated(min_mip, small_phi_time)

    log.info("Finished calculating big-phi data for {}.".format(subsystem))
    log.debug("RESULT: \n" + str(result))

    return result


def _big_mip_cache_key(subsystem):
    """The cache key of the subsystem.

    This includes the native hash of the subsystem and all configuration values
    which change the results of ``big_mip``.
    """
    return (
        hash(subsystem),
        config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS,
        config.CUT_ONE_APPROXIMATION,
        config.MEASURE,
        config.PRECISION,
        config.VALIDATE_SUBSYSTEM_STATES,
        config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI
    )


# Wrapper to ensure that the cache key is the native hash of the subsystem, so
# joblib doesn't mistakenly recompute things when the subsystem's MICE cache is
# changed. The cache is also keyed on configuration values which affect the
# value of the computation.
@functools.wraps(_big_mip)
def big_mip(subsystem):
    return _big_mip(_big_mip_cache_key(subsystem), subsystem)


def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return big_mip(subsystem).phi


def subsystems(network, state):
    """Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible state.
    """
    validate.is_network(network)

    for subset in utils.powerset(network.node_indices):
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            pass


def all_complexes(network, state):
    """Return a generator for all complexes of the network.

    Includes reducible, zero-phi complexes (which are not, strictly speaking,
    complexes at all).
    """
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
        state (tuple[int]): The state of the network.

    Yields:
        Subsystem: The next subsystem which could be a complex.
    """
    validate.is_network(network)

    causally_significant_nodes = utils.causally_significant_nodes(network.cm)

    for subset in utils.powerset(causally_significant_nodes):
        # Don't return empty system
        if len(subset) == 0:
            continue

        # Don't return subsystems that are in an impossible state.
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            continue


def complexes(network, state):
    """Return all irreducible complexes of the network."""
    return tuple(filter(None, (big_mip(subsystem) for subsystem in
                               possible_complexes(network, state))))


def main_complex(network, state):
    """Return the main complex of the network."""
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
