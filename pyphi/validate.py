#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate
~~~~~~~~

Methods for validating common types of input.
"""

import numpy as np

from collections import Iterable
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
    return True


def nodelist(nodes, name):
    if not isinstance(nodes, Iterable):
        raise ValueError("{} must be an iterable.".format(name))
    if not all(isinstance(node, Node) for node in nodes):
        raise ValueError("{} must consist only of Nodes (perhaps you "
                         "gave a node's index instead?)".format(name))
    if not isinstance(nodes, tuple):
        nodes = tuple(nodes)
    return nodes


def tpm(tpm):
    if (tpm.shape[-1] != len(tpm.shape) - 1):
        raise ValueError(
            "Invalid TPM: There must be a dimension for each node, "
            "each one being the size of the corresponding node's "
            "state space, plus one dimension that is the same size "
            "as the network.")
    # TODO extend to nonbinary nodes
    if (tpm.shape[:-1] != tuple([2] * tpm.shape[-1])):
        raise ValueError(
            "Invalid TPM: We can only handle binary nodes at the "
            "moment. Each dimension except the last must be of size "
            "2.")
    return True


def connectivity_matrix(cm):
    if (cm.ndim != 2):
        raise ValueError("Connectivity matrix must be 2-dimensional.")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if not np.all(np.logical_or(cm == 1, cm == 0)):
        raise ValueError("Connectivity matrix must contain only binary "
                         "values.")
    return True


# TODO test
def _state_reachable(current_state, tpm):
    """Return whether a state can be reached according to the given TPM."""
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    test = tpm - np.array(current_state)
    return np.any(np.logical_and(-1 < test, test < 1).all(-1))


# TODO test
def _state_reachable_from(past_state, current_state, tpm):
    """Return whether a state is reachable from the given past state."""
    test = tpm[tuple(past_state)] - np.array(current_state)
    return np.all(np.logical_and(-1 < test, test < 1))


# TODO test
def state(network):
    """Validate a network's current and past state."""
    current_state, past_state = network.current_state, network.past_state
    tpm = network.tpm
    # Check that the current and past states are the right size.
    invalid_state = False
    if len(current_state) != network.size:
        invalid_state = ('current', len(network.current_state))
    if len(past_state) != network.size:
        invalid_state = ('past', len(network.past_state))
    if invalid_state:
        raise ValueError("Invalid {} state: there must be one entry per node "
                         "in the network; this state has {} entries, but "
                         "there are {} nodes.".format(invalid_state[0],
                                                      invalid_state[1],
                                                      network.size))
    # Check that the current state is reachable from some state.
    if not _state_reachable(current_state, tpm):
        raise StateUnreachableError(
            current_state, past_state, tpm,
            "The current state is unreachable according to the given TPM.")
    # Check that the current state is reachable from the given past state.
    if not _state_reachable_from(past_state, current_state, tpm):
        raise StateUnreachableError(
            current_state, past_state, tpm,
            "The current state cannot be reached from the past state "
            "according to the given TPM.")
    return True


# TODO test
def network(network):
    """Validate TPM, connectivity matrix, and current and past state."""
    tpm(network.tpm)
    state(network)
    connectivity_matrix(network.connectivity_matrix)
    if network.connectivity_matrix.shape[0] != network.size:
        raise ValueError("Connectivity matrix must be NxN, where N is the "
                         "number of nodes in the network.")
    return True
