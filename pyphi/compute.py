#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute.py

"""
Methods for computing concepts, constellations, and integrated information of
subsystems.
"""

import functools
import logging
import multiprocessing
from time import time

import numpy as np

from . import config, constants, memory, utils, validate
from .concept_caching import concept as _concept
from .config import PRECISION
from .models import BigMip, Constellation, Cut
from .network import Network
from .subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)


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

    .. note::
        The output can be persistently cached to avoid recomputation. This may
        be enabled in the configuration file---however, it is only available if
        the caching backend is a database (not the filesystem). See the
        documentation for the |concept_caching| and |config| modules.
    """
    start = time()

    def time_annotated(concept):
        concept.time = time() - start
        return concept

    # Pre-checks:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If the mechanism is empty, there is no concept.
    if not mechanism:
        return time_annotated(subsystem.null_concept)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Passed prechecks; pass it over to the concept caching logic if enabled.
    # Concept caching is only available if the caching backend is a database.
    if (config.CACHE_CONCEPTS and
            config.CACHING_BACKEND == constants.DATABASE):
        return time_annotated(_concept(
            subsystem, mechanism, purviews=purviews,
            past_purviews=past_purviews, future_purviews=future_purviews))
    else:
        return time_annotated(subsystem.concept(
            mechanism, purviews=purviews, past_purviews=past_purviews,
            future_purviews=future_purviews))


def _sequential_constellation(subsystem, mechanisms=False, purviews=False,
                              past_purviews=False, future_purviews=False):
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)
    concepts = [concept(subsystem, mechanism, purviews=purviews,
                        past_purviews=past_purviews,
                        future_purviews=future_purviews)
                for mechanism in mechanisms]
    # Filter out falsy concepts, i.e. those with effectively zero Phi.
    return Constellation(filter(None, concepts))


def _concept_wrapper(in_queue, out_queue, subsystem, purviews=False,
                     past_purviews=False, future_purviews=False):
    """Wrapper for parallel evaluation of concepts."""
    while True:
        (index, mechanism) = in_queue.get()
        if mechanism is None:
            break
        new_concept = concept(subsystem, mechanism, purviews=purviews,
                              past_purviews=past_purviews,
                              future_purviews=future_purviews)
        if new_concept.phi > 0:
            out_queue.put(new_concept)
    out_queue.put(None)


# TODO: can return a negative number if NUMBER_OF_CORES
# is too negative. Handle this
def get_num_processes():
    """Return the number of processes to use in parallel."""
    cpu_count = multiprocessing.cpu_count()

    if config.NUMBER_OF_CORES == 0:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value may not be 0.')

    if config.NUMBER_OF_CORES > cpu_count:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value must be less than or '
            'equal to the available number of cores ({} for this '
            'system).'.format(cpu_count))

    if config.NUMBER_OF_CORES < 0:
        return cpu_count + config.NUMBER_OF_CORES + 1

    return config.NUMBER_OF_CORES


def _parallel_constellation(subsystem, mechanisms=False, purviews=False,
                            past_purviews=False, future_purviews=False):
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices)

    number_of_processes = get_num_processes()

    # Define input and output queues.
    # Load the input queue with all possible cuts and a 'poison pill' for each
    # process.
    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    for i, mechanism in enumerate(mechanisms):
        in_queue.put((i, mechanism))
    for i in range(number_of_processes):
        in_queue.put((None, None))
    # Initialize the processes and start them.
    processes = [
        multiprocessing.Process(target=_concept_wrapper,
                                args=(in_queue, out_queue, subsystem, purviews,
                                      past_purviews, future_purviews))
        for i in range(number_of_processes)
    ]
    for i in range(number_of_processes):
        processes[i].start()
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
    return Constellation(concepts)


_constellation_doc = \
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
_sequential_constellation.__doc__ = _constellation_doc
_parallel_constellation.__doc__ = _constellation_doc
# TODO fix and release in version 0.7.0
# if config.PARALLEL_CONCEPT_EVALUATION:
#     constellation = _parallel_constellation
# else:
#     constellation = _sequential_constellation
constellation = _sequential_constellation


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
    return round(ci, PRECISION)


# TODO document
def _null_bigmip(subsystem):
    """Return a |BigMip| with zero |big_phi| and empty constellations.

    This is the MIP associated with a reducible subsystem.
    """
    return BigMip(subsystem=subsystem, cut_subsystem=subsystem, phi=0.0,
                  unpartitioned_constellation=(), partitioned_constellation=())


def _single_node_mip(subsystem):
    """Return a |BigMip| of a single-node with a selfloop.

    Whether these have a nonzero |Phi| value depends on the PyPhi constants.
    """
    if config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI:
        # TODO return the actual concept
        return BigMip(
            phi=0.5,
            unpartitioned_constellation=(),
            partitioned_constellation=(),
            subsystem=subsystem,
            cut_subsystem=subsystem)
    else:
        return _null_bigmip(subsystem)


def _evaluate_cut(uncut_subsystem, cut, unpartitioned_constellation):
    """Find the |BigMip| for a given cut."""
    log.debug("Evaluating cut {}...".format(cut))

    cut_subsystem = Subsystem(uncut_subsystem.network,
                              uncut_subsystem.state,
                              uncut_subsystem.node_indices,
                              cut=cut,
                              mice_cache=uncut_subsystem._mice_cache)
    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = set([c.mechanism for c in unpartitioned_constellation])
    else:
        mechanisms = set(
            [c.mechanism for c in unpartitioned_constellation] +
            list(cut.all_cut_mechanisms(uncut_subsystem.node_indices)))
    partitioned_constellation = constellation(cut_subsystem, mechanisms)

    log.debug("Finished evaluating cut {}.".format(cut))

    phi = constellation_distance(unpartitioned_constellation,
                                 partitioned_constellation)

    return BigMip(
        phi=round(phi, PRECISION),
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=partitioned_constellation,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


# Wrapper for _evaluate_cut for parallel processing.
def _eval_wrapper(in_queue, out_queue, subsystem, unpartitioned_constellation):
    while True:
        cut = in_queue.get()
        if cut is None:
            break
        new_mip = _evaluate_cut(subsystem, cut, unpartitioned_constellation)
        out_queue.put(new_mip)
    out_queue.put(None)


def _find_mip_parallel(subsystem, cuts, unpartitioned_constellation, min_mip):
    """Find the MIP for a subsystem with a parallel loop over all cuts.

    Uses the specified number of cores.
    """
    number_of_processes = get_num_processes()
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
        elif utils.phi_eq(new_mip.phi, 0):
            min_mip = new_mip
            for process in processes:
                process.terminate()
            break
        else:
            if new_mip < min_mip:
                min_mip = new_mip
    return min_mip


def _find_mip_sequential(subsystem, cuts, unpartitioned_constellation,
                         min_mip):
    """Find the minimal cut for a subsystem by sequentially loop over all cuts.

    Holds only two |BigMip|s in memory at once.
    """
    for i, cut in enumerate(cuts):
        new_mip = _evaluate_cut(subsystem, cut, unpartitioned_constellation)
        log.debug("Finished {} of {} cuts.".format(
            i + 1, len(cuts)))
        if new_mip < min_mip:
            min_mip = new_mip
        # Short-circuit as soon as we find a MIP with effectively 0 phi.
        if not min_mip:
            break
    return min_mip


def big_mip_bipartitions(nodes):
    """Return all |big_phi| cuts for the given nodes.

    This value changes based on `config.CUT_ONE_APPROXIMATION`.

    Args:
        nodes tuple(int): The node indices to partition.
    Returns:
        list(Cut): All unidirectional partitions.
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
        big_mip (|BigMip|): A nested structure containing all the data from the
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
        return time_annotated(_single_node_mip(subsystem))

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
    """
    inputs = np.sum(network.connectivity_matrix, 0)
    outputs = np.sum(network.connectivity_matrix, 1)
    nodes_have_inputs_and_outputs = np.logical_and(inputs > 0, outputs > 0)
    causally_significant_nodes = np.where(nodes_have_inputs_and_outputs)[0]
    for subset in utils.powerset(causally_significant_nodes):
        # Don't return subsystems that are in an impossible state.
        try:
            yield Subsystem(network, state, subset)
        except validate.StateUnreachableError:
            pass


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
