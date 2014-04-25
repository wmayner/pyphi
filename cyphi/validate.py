#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate
~~~~~~~~

Methods for validating common types of input.
"""

import numpy as np

from collections import Iterable
from .models import Cut
from .node import Node
from . import constants


class StateUnreachableError(ValueError):
    """Raised when the current state of a network cannot be reached, either
    from any state or from a given past state."""

    def __init__(self, current_state, past_state, tpm, message):
        self.current_state = current_state
        self.past_state = past_state
        self.tpm = tpm
        self.message = message

    def __str__(self):
        return self.message


def direction(direction):
    if direction not in constants.DIRECTIONS:
        raise ValueError("Direction must be either 'past' or 'future'.")


def nodelist(nodes, name):
    if not isinstance(nodes, Iterable):
        raise ValueError("{} must be an iterable.".format(name))
    if not all(isinstance(node, Node) for node in nodes):
        raise ValueError("{} must consist only of Nodes (perhaps you "
                         "gave a node's index instead?)".format(name))
    if not isinstance(nodes, tuple):
        nodes = tuple(nodes)
    return nodes


def cut(subsystem, partition):
    severed, intact = partition[0], partition[1]
    # Convert single nodes to singleton tuples and iterables to tuples.
    if not isinstance(severed, Iterable):
        severed = (severed,)
    else:
        severed = tuple(severed)
    if not isinstance(intact, Iterable):
        intact = (intact,)
    else:
        intact = tuple(intact)
    # Validate.
    if not (len(subsystem.nodes) == len(severed + intact) and
            set(subsystem.nodes) == set(severed + intact)):
        raise ValueError("Each node in the subsystem must appear exactly once "
                         "in the partition.")
    return Cut(severed, intact)


def tpm(tpm):
    if (tpm.shape[-1] != len(tpm.shape) - 1):
        raise ValueError("Invalid TPM: There must be a dimension for each node, "
                         "each one being the size of the corresponding node's "
                         "state space, plus one dimension that is the same size "
                         "as the network.")
    # TODO extend to nonbinary nodes
    if (tpm.shape[:-1] != tuple([2] * tpm.shape[-1])):
        raise ValueError("Invalid TPM: We can only handle binary nodes at the "
                         "moment. Each dimension except the last must be of size "
                         "2.")


# TODO! test
def connectivity_matrix(network):
    cm = network.connectivity_matrix
    if cm is not None:
        if (cm.ndim != 2):
            raise ValueError("Connectivity matrix must be 2-dimensional.")
        if cm.shape[0] != cm.shape[1]:
            raise ValueError("Connectivity matrix must be square.")
        if cm.shape[0] != network.size:
            raise ValueError("Connectivity matrix must be NxN, where N is the "
                             "number of nodes in the network.")


# TODO test
def _state_reachable(current_state, tpm):
    """Return whether a state can be reached according to the given TPM."""
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    test = tpm - current_state
    return np.any(np.logical_and(-1 < test, test < 1).all(1))


# TODO test
def _state_reachable_from(past_state, current_state, tpm):
    """Return whether a state is reachable from the given past state."""
    test = tpm[past_state] - current_state
    return np.all(np.logical_and(-1 < test, test < 1))


# TODO test
def state(network):
    """Validate a network's current and past state."""
    current_state, past_state = network.current_state, network.past_state
    tpm = network.tpm
    # Check that the state is the right size.
    if (len(current_state) != network.size or len(past_state) != network.size):
        raise ValueError("Invalid state: there must be one entry per node in "
                         "the network; this state has {} entries, but there "
                         "are {} nodes.".format(network.current_state.size,
                                                network.size))
    # Check that the state is reachable from some state.
    if not _state_reachable(current_state, tpm):
        raise StateUnreachableError(current_state, past_state, tpm,
            "The current state is unreachable according to the given TPM.")
    # Check that the state is reachable from the given past state.
    if not _state_reachable_from(past_state, current_state, tpm):
        raise StateUnreachableError(current_state, past_state, tpm,
            "The current state cannot be reached from the past state "
            "according to the given TPM.")


# TODO test
def network(network):
    """Validate TPM, connectivity matrix, and current and past state."""
    tpm(network.tpm)
    connectivity_matrix(network)
    state(network)
