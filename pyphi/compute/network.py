#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/network.py

"""
Functions for computing network-level properties.
"""

import logging

from .. import config, exceptions, utils, validate
from ..models import _null_sia
from ..subsystem import Subsystem
from .parallel import MapReduce
from .subsystem import sia

# Create a logger for this module.
log = logging.getLogger(__name__)


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

    .. note::
        Does not return subsystems that are in an impossible state (after
        conditioning the subsystem TPM on the state of the other nodes).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        Subsystem: A |Subsystem| for each subset of nodes in the network,
        excluding subsystems that would be in an impossible state.
    """
    return _reachable_subsystems(network, network.node_indices, state)


def possible_complexes(network, state):
    """Return a generator of subsystems of a network that could be a complex.

    This is the just powerset of the nodes that have at least one input and
    output (nodes with no inputs or no outputs cannot be part of a main
    complex, because they do not have a causal link with the rest of the
    subsystem in the previous or next timestep, respectively).

    .. note::
        Does not return subsystems that are in an impossible state (after
        conditioning the subsystem TPM on the state of the other nodes).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        Subsystem: The next subsystem that could be a complex.
    """
    return _reachable_subsystems(
        network, network.causally_significant_nodes, state)


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

    .. note::
        Includes reducible, zero-|big_phi| complexes (which are not, strictly
        speaking, complexes at all).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        SystemIrreducibilityAnalysis: A |SIA| for each |Subsystem| of the
        |Network|.
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
    """Return all irreducible complexes of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        SystemIrreducibilityAnalysis: A |SIA| for each |Subsystem| of the
        |Network|, excluding those with |big_phi = 0|.
    """
    engine = FindIrreducibleComplexes(possible_complexes(network, state))
    return engine.run(config.PARALLEL_COMPLEX_EVALUATION)


def major_complex(network, state):
    """Return the major complex of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        SystemIrreducibilityAnalysis: The |SIA| for the |Subsystem| with
        maximal |big_phi|.
    """
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
    """Return a list of maximal non-overlapping complexes.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        list[SystemIrreducibilityAnalysis]: A list of |SIA| for non-overlapping
        complexes with maximal |big_phi| values.
    """
    result = []
    covered_nodes = set()

    for c in reversed(sorted(complexes(network, state))):
        if not any(n in covered_nodes for n in c.subsystem.node_indices):
            result.append(c)
            covered_nodes = covered_nodes | set(c.subsystem.node_indices)

    return result
