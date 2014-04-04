#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains methods for validating common types of input."""

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
        raise ValueError("Each node in the subsystem must appear exactly" +
                            " once in the partition.")
    return Cut(severed, intact)
