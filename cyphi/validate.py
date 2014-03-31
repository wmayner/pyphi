#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Iterable
from .node import Node


class Validate:

    """Contains methods for validating common types of input."""

    @staticmethod
    def direction(direction):
        if direction != 'past' and direction != 'future':
            raise ValueError("Direction must be either 'past' or 'future'.")

    @staticmethod
    def nodelist(nodes, name):
        if not isinstance(nodes, Iterable):
            raise ValueError(name + " must be an iterable.")
        if not all(isinstance(node, Node) for node in nodes):
            raise ValueError(name + " must consist only of Nodes (perhaps " +
                            "you gave a node's index instead?)")
