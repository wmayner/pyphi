#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute
~~~~~~~

Methods for computing concepts, constellations, and integrated information of
subsystems.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components

from .models import Cut, BigMip
from . import constants
from . import utils


def concept_distance(c1, c2):
    """Return the distance between two concepts in concept-space."""
    return sum([utils.hamming_emd(c1.location[constants.PAST],
                                  c2.location[constants.PAST]),
                utils.hamming_emd(c1.location[constants.FUTURE],
                                  c2.location[constants.FUTURE])])


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


# TODO Figure out null concept - should it be a param for each one?
def constellation_distance(C1, C2, null_concept):
    """Return the distance between two constellations in concept-space."""
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
    return constellation_distance(subsystem.constellation(), ())


# TODO make null concept an attribute (property)?
def _evaluate_cut(subsystem, partition, unpartitioned_constellation):
    # Compute forward mip
    forward_cut = Cut(partition[0], partition[1])
    forward_constellation = subsystem.constellation(forward_cut)
    forward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   forward_constellation,
                                   subsystem.null_concept()),
        partition=forward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=forward_constellation,
        subsystem=subsystem)
    # Compute backward mip
    backward_cut = Cut(partition[1], partition[0])
    backward_constellation = subsystem.constellation(backward_cut)
    backward_mip = BigMip(
        phi=constellation_distance(unpartitioned_constellation,
                                   backward_constellation,
                                   subsystem.null_concept()),
        partition=backward_cut,
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=backward_constellation,
        subsystem=subsystem)
    # Choose minimal unidirectional cut
    mip = min(forward_mip, backward_mip)
    # Return the mip if the subsystem with the given partition is not reducible
    return mip if mip.phi > constants.EPSILON else None


def big_mip(subsystem):
    """Return the MIP for a subsystem."""
    # If the subsystem is not strongly connected, phi is necessarily zero, so
    # we immediately return a null MIP
    cm = subsystem.network.connectivity_matrix
    num_components, _ = connected_components(cm) if cm != None else (1, None)
    if num_components > 1:
        return BigMip(subsystem=subsystem,
                      phi=0.0,
                      # TODO should this be null cut?
                      partition=subsystem.null_cut,
                      unpartitioned_constellation=[],
                      partitioned_constellation=[])

    # Calculate the unpartitioned constellation
    unpartitioned_constellation = subsystem.constellation(subsystem.null_cut)
    # The first bipartition is the null cut, so skip it
    bipartitions = list(utils.bipartition(subsystem.nodes))[1:]
    # Parallel loop over all partitions
    mip_candidates = Parallel(n_jobs=constants.NUMBER_OF_CORES)(
        delayed(_evaluate_cut)(subsystem, partition,
                               unpartitioned_constellation)
        for partition in bipartitions)
    return min(mip_candidates) if any(mip_candidates) else None


def big_phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return big_mip(subsystem).phi
