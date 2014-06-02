#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute
~~~~~~~

Methods for computing concepts, constellations, and integrated information of
subsystems.
"""

import functools
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from . import utils, options
from .models import Concept, Cut, BigMip, MarblSet
from .network import Network
from .subsystem import Subsystem
from .constants import PAST, FUTURE, MAXMEM, memory
from .lru_cache import lru_cache


@memory.cache(ignore=['subsystem', 'mechanism', 'cut'])
def _concept(cache_key, subsystem, mechanism, cut):
    # If any node in the mechanism either has no inputs from the subsystem or
    # has no outputs to the subsystem, then the mechanism is necessarily
    # reducible and cannot be a concept (since removing that node would make no
    # difference to at least one of the MICEs).
    if not (subsystem._all_connect_to_any(mechanism, subsystem.nodes) and
            subsystem._any_connect_to_all(subsystem.nodes, mechanism)):
        return None

    past_mice = subsystem.core_cause(mechanism, cut)
    future_mice = subsystem.core_effect(mechanism, cut)
    phi = min(past_mice.phi, future_mice.phi)

    if phi < options.EPSILON:
        return None
    return Concept(
        mechanism=mechanism,
        location=np.array([
            subsystem.expand_cause_repertoire(past_mice.mechanism,
                                              past_mice.purview,
                                              past_mice.repertoire, cut),
            subsystem.expand_effect_repertoire(future_mice.mechanism,
                                               future_mice.purview,
                                               future_mice.repertoire, cut)]),
        phi=phi,
        cause=past_mice,
        effect=future_mice)


def concept(subsystem, mechanism, cut=None):
    """Return the concept specified by the a mechanism within a subsytem.

    Args:
        subsystem (Subsytem): The context in which the mechanism should be
            considered.
        mechanism (tuple(Node)): The candidate set of nodes.

    Keyword Args:
        cut (Cut): The optional unidirectional cut that should be applied to
            the network when doing the calculation. Defaults to ``None``, where
            no cut is applied.

    Returns:
        ``Concept`` or ``None`` -- The pair of maximally irreducible
            cause/effect repertoires that constitute the concept specified by
            the given mechanism, or ``None`` if there isn't one.

    .. note::
        The output is "persistently cached" (saved to the disk for later
        access), to avoid recomputation. The cache key is the hash of the
        normal form of the multiset of the mechanism nodes' Markov blankets
        (not the mechanism itself). This results in more cache hits, since the
        output depends only on the causual properties of the nodes. See the
        documentation for the `marbl specification
        <https://github.com/wmayner/marbl>`_, and the `marbl-python
        implementation <http://pythonhosted.org/marbl-python/>`_.
    """
    # Generate the cache key for memoizing concepts
    if cut is None:
        cut = subsystem.null_cut
    cache_key = hash(MarblSet(mechanism, cut))
    # Pass on the cache key
    return _concept(cache_key, subsystem, mechanism, cut)


def constellation(subsystem, cut=None):
    """Return the conceptual structure of this subsystem.

    Args:
        subsystem (Subsytem): The subsystem for which to determine the
            constellation.

    Keyword Args:
        cut (Cut): The optional unidirectional cut that should be applied to
            the network when doing the calculation. Defaults to ``None``, where
            no cut is applied.

    Returns:
        ``tuple(Concept)`` -- A tuple of all the Concepts in the constellation.
    """
    concepts = [concept(subsystem, mechanism, cut) for mechanism in
                utils.powerset(subsystem.nodes)]
    # Filter out non-concepts
    return tuple(filter(None, concepts))


@lru_cache(maxmem=MAXMEM)
def concept_distance(c1, c2):
    """Return the distance between two concepts in concept-space.

    Args:
        c1 (Mice): The first concept.
        c2 (Mice): The second concept.

    Returns:
        ``float`` -- The distance between the two concepts in concept-space.
    """
    return sum([utils.hamming_emd(c1.location[PAST],
                                  c2.location[PAST]),
                utils.hamming_emd(c1.location[FUTURE],
                                  c2.location[FUTURE])])


def _constellation_distance_simple(C1, C2, null_concept):
    """Return the distance between two constellations in concept-space,
    assuming the only difference between them is that some concepts have
    disappeared."""
    # Make C1 refer to the bigger constellation
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c for c in C1 if c not in C2]
    return sum(c.phi * concept_distance(c, null_concept) for c in destroyed)


def _constellation_distance_emd(C1, C2, unique_C1, unique_C2, null_concept):
    """Return the distance between two constellations in concept-space,
    using the generalized EMD."""
    shared_concepts = [c for c in C1 if c in C2]
    # Construct null concept and list of all unique concepts.
    all_concepts = shared_concepts + unique_C1 + unique_C2 + [null_concept]
    # Construct the two phi distributions.
    d1, d2 = [[c.phi if c in constellation else 0 for c in all_concepts]
              for constellation in (C1, C2)]
    # Calculate how much phi disappeared and assign it to the null concept
    # (the null concept is the last element in the distribution).
    residual = sum(d1) - sum(d2)
    if residual > 0:
        d2[-1] = residual
    if residual < 0:
        d1[-1] = residual
    # Generate the ground distance matrix.
    distance_matrix = np.array([
        [concept_distance(i, j) for i in all_concepts] for j in
        all_concepts])

    return utils.emd(np.array(d1), np.array(d2), distance_matrix)


@lru_cache(maxmem=MAXMEM)
def constellation_distance(C1, C2, null_concept):
    """Return the distance between two constellations in concept-space.

    Args:
        C1 (tuple(Concept)): The first constellation.
        C2 (tuple(Concept)): The second constellation.
        null_concept (Concept): The null concept of a candidate set, *i.e* the
            "origin" of the concept space in which the given constellations
            reside.

    Returns:
        ``float`` -- The distance between the two constellations in
        concept-space.
    """
    concepts_only_in_C1 = [c for c in C1 if c not in C2]
    concepts_only_in_C2 = [c for c in C2 if c not in C1]
    # If the only difference in the constellations is that some concepts
    # disappeared, then we don't need to use the emd.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        return _constellation_distance_simple(C1, C2, null_concept)
    else:
        return _constellation_distance_emd(C1, C2,
                                           concepts_only_in_C1,
                                           concepts_only_in_C2,
                                           null_concept)


# TODO Define this for cuts? need to have a cut in the null concept then
def conceptual_information(subsystem):
    """Return the conceptual information for a subsystem.

    This is the distance from the subsystem's constellation to the null
    concept."""
    return constellation_distance(constellation(subsystem), ())


# TODO document
def _null_mip(subsystem):
    """Returns a BigMip with zero phi and empty constellations.

    This is the MIP associated with a reducible subsystem."""
    return BigMip(subsystem=subsystem,
                  phi=0.0,
                  cut=subsystem.null_cut,
                  unpartitioned_constellation=[], partitioned_constellation=[])


def _single_node_mip(subsystem):
    """Returns a the BigMip of a single-node with a selfloop.

    Whether these have a nonzero |Phi| value depends on the CyPhi options."""
    if options.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI:
        # TODO return the actual concept
        return BigMip(
            phi=0.5,
            cut=Cut(subsystem.nodes, subsystem.nodes),
            unpartitioned_constellation=None,
            partitioned_constellation=None,
            subsystem=subsystem)
    else:
        return _null_mip(subsystem)


# TODO document
def _evaluate_cut(subsystem, partition, unpartitioned_constellation):
    # Compute forward mip.
    forward_cut = Cut(partition[0], partition[1])
    forward_constellation = constellation(subsystem, cut=forward_cut)
    forward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   forward_constellation,
                                   subsystem.null_concept),
        cut=forward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=forward_constellation,
        subsystem=subsystem)
    # Compute backward mip.
    backward_cut = Cut(partition[1], partition[0])
    backward_constellation = constellation(subsystem, cut=backward_cut)
    backward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   backward_constellation,
                                   subsystem.null_concept),
        cut=backward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=backward_constellation,
        subsystem=subsystem)
    # Choose minimal unidirectional cut.
    mip = min(forward_mip, backward_mip)
    # Return the mip if the subsystem with the given partition is not
    # reducible.
    return mip if mip.phi > options.EPSILON else _null_mip(subsystem)


# TODO document big_mip
@memory.cache(ignore=['subsystem'])
def _big_mip(subsystem, cache_key):
    """Return the MIP of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        ``BigMip`` -- A nested structure containing all the data from the
        intermediate calculations. The top level contains the basic MIP
        information for the given subsystem. See :class:`models.BigMip`.
    """
    # Special case for single-node subsystems.
    if (len(subsystem.nodes) == 1):
        return _single_node_mip(subsystem)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty; or
    #   - an elementary mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null MIP.

    if not subsystem:
        return _null_mip(subsystem)

    if subsystem.network.connectivity_matrix is not None:
        # Get the connectivity of just the subsystem nodes.
        submatrix_indices = np.ix_([node.index for node in subsystem.nodes],
                                   [node.index for node in subsystem.nodes])
        cm = subsystem.network.connectivity_matrix[submatrix_indices]
        # Get the number of strongly connected components.
        num_components, _ = connected_components(csr_matrix(cm))
        if num_components > 1:
            return _null_mip(subsystem)

    # The first bipartition is the null cut (trivial bipartition), so skip it.
    bipartitions = utils.bipartition(subsystem.nodes)[1:]
    if not bipartitions:
        return _null_mip(subsystem)

    # =========================================================================

    # Calculate the unpartitioned constellation.
    unpartitioned_constellation = constellation(subsystem, subsystem.null_cut)
    # Parallel loop over all partitions (use all but one CPU).
    mip_candidates = Parallel(n_jobs=(-2 if options.PARALLEL_CUT_EVALUATION
                                      else 1),
                              verbose=options.VERBOSE_PARALLEL)(
        delayed(_evaluate_cut)(subsystem,
                               partition,
                               unpartitioned_constellation)
        for partition in bipartitions)

    return min(mip_candidates)


# Wrapper so that joblib.Memory caches by the native hash
@functools.wraps(_big_mip)
def big_mip(subsystem):
    return _big_mip(subsystem, hash(subsystem))


@lru_cache(maxmem=MAXMEM)
def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return big_mip(subsystem).phi


@lru_cache(maxmem=MAXMEM)
def complexes(network):
    """Return a generator for all complexes of the network.

    This includes reducible, zero-phi complexes (which are not, strictly
    speaking, complexes at all)."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return (big_mip(subsystem) for subsystem in network.subsystems())


@lru_cache(maxmem=MAXMEM)
def main_complex(network):
    """Return the main complex of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return max(complexes(network))


def subsystems(network):
    """Return a generator of all possible subsystems of a network.

    This is the just powerset of the network's set of nodes."""
    for subset in utils.powerset(network.nodes):
        yield Subsystem(subset, network.current_state, network.past_state,
                        network)
