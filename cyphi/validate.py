#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate
~~~~~~~~

Methods for validating common types of input.
"""

from collections import Iterable
from .models import Cut
from .node import Node
from . import constants


def direction(direction):
    if direction not in constants.DIRECTIONS:
        raise ValueError("Direction must be either 'past' or 'future'.")


def nodelist(nodes, name):
    if not isinstance(nodes, Iterable):
        raise ValueError(name + " must be an iterable.")
    if not all(isinstance(node, Node) for node in nodes):
        raise ValueError(name + " must consist only of Nodes (perhaps you " +
                         "gave a node's index instead?)")


def cut(subsystem, partition):
    severed, intact = partition[0], partition[1]
    # Convert single nodes to singleton tuples and iterables to tuples
    if not isinstance(severed, Iterable):
        severed = (severed,)
    else:
        severed = tuple(severed)
    if not isinstance(intact, Iterable):
        intact = (intact,)
    else:
        intact = tuple(intact)
    # Validate
    if not (len(subsystem.nodes) == len(severed + intact) and
            set(subsystem.nodes) == set(severed + intact)):
        raise ValueError("Each node in the subsystem must appear exactly once \
                         in the partition.")
    return Cut(severed, intact)


def tpm(tpm):
    if (tpm.shape[-1] != len(tpm.shape) - 1):
        raise ValueError(
            """Invalid TPM: There must be a dimension for each node, each one
            being the size of the corresponding node's state space, plus one
            dimension that is the same size as the network.""")
    # TODO extend to nonbinary nodes
    if (tpm.shape[:-1] != tuple([2] * tpm.shape[-1])):
        raise ValueError(
            """Invalid TPM: We can only handle binary nodes at the moment. Each
            dimension except the last must be of size 2.""")


# TODO! test
def connectivity_matrix(network):
    cm = network.connectivity_matrix
    if cm != None:
        if (cm.ndim != 2):
            raise ValueError("Connectivity matrix must be 2-dimensional.")
        if cm.shape[0] != cm.shape[1]:
            raise ValueError("Connectivity matrix must be square.")
        if cm.shape[0] != network.size:
            raise ValueError("Connectivity matrix must be NxN, where N is the \
                             number of nodes in the network.")


def state(network):
    if (network.current_state.size != network.size
            or network.past_state.size != network.size):
        raise ValueError(
            "Invalid state: there must be one entry per node in the network; \
            this state has " + str(network.current_state.size) + " entries, \
            but there \ are " + str(network.size) + " nodes.")

def network(network):
    """Validate TPM, connectivity_matrix, and current and past state."""
    tpm(network.tpm)
    connectivity_matrix(network)
    state(network)
