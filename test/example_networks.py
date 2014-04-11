#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from cyphi.network import Network
from cyphi.subsystem import Subsystem

# TODO pass just the subsystem (contains a reference to the network)


def m():
    """Matlab default network.

    Comes with subsystems attached, no assembly required.

    Diagram:

    |           +~~~~~~+
    |    +~~~~~>|   A  |<~~~~+
    |    |      | (OR) +~~~+ |
    |    |      +~~~~~~+   | |
    |    |                 | |
    |    |                 v |
    |  +~+~~~~~~+      +~~~~~+~+
    |  |   B    |<~~~~~+   C   |
    |  | (COPY) +~~~~~>| (XOR) |
    |  +~~~~~~~~+      +~~~~~~~+

    TPM:

    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    | Past state ~~> Current state |
    |~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~|
    |   A, B, C    |    A, B, C    |
    |~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~|
    |  {0, 0, 0}   |   {0, 0, 0}   |
    |  {0, 0, 1}   |   {1, 1, 0}   |
    |  {0, 1, 0}   |   {1, 0, 1}   |
    |  {0, 1, 1}   |   {1, 1, 1}   |
    |  {1, 0, 0}   |   {0, 0, 1}   |
    |  {1, 0, 1}   |   {1, 1, 1}   |
    |  {1, 1, 0}   |   {1, 0, 0}   |
    |  {1, 1, 1}   |   {1, 1, 0}   |
    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    """
    # TODO? make these into dictionaries/named tuples
    current_state = np.array([1, 0, 0])
    past_state = np.array([1, 1, 0])
    tpm = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]]).reshape([2] * 3 + [3], order="F").astype(float)
    m = Network(tpm, current_state, past_state)
    m.subsys_n0n2 = Subsystem([m.nodes[0], m.nodes[2]],
                              m.current_state,
                              m.past_state,
                              m)
    m.subsys_n1n2 = Subsystem([m.nodes[1], m.nodes[2]],
                              m.current_state,
                              m.past_state,
                              m)
    m.subsys_all = Subsystem(m.nodes, m.current_state, m.past_state, m)
    return m


def s():
    """ Simple 'AND' network.

    Diagram:

    |        +~~~+
    |    +~~>| A |<~~+
    |    |   +~~~+   |
    |    |    AND    |
    |  +~+~+       +~+~+
    |  | B |       | C |
    |  +~~~+       +~~~+

    TPM:

    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    |  Past state ~~> Current state |
    |~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~~|
    |   A, B, C    |    A, B, C     |
    |~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~~|
    |  {0, 0, 0}   |   {0, 0, 0}    |
    |  {0, 0, 1}   |   {0, 0, 0}    |
    |  {0, 1, 0}   |   {0, 0, 0}    |
    |  {0, 1, 1}   |   {1, 0, 0}    |
    |  {1, 0, 0}   |   {0, 0, 0}    |
    |  {1, 0, 1}   |   {0, 0, 0}    |
    |  {1, 1, 0}   |   {0, 0, 0}    |
    |  {1, 1, 1}   |   {0, 0, 0}    |
    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    """
    a_just_turned_on = np.array([1, 0, 0])
    a_about_to_be_on = np.array([0, 1, 1])
    tpm = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]).reshape([2] * 3 + [3]).astype(float)

    s = Network(tpm, a_just_turned_on, a_about_to_be_on)
    # Subsystem([n0, n1, n2]) of the simple 'AND' network.
    # Node n0 (A in the diagram) has just turned on.
    s.subsys_all_a_just_on = Subsystem(s.nodes,
                                       s.current_state,
                                       s.past_state,
                                       s)
    # Subsystem([n0, n1, n2]) of the simple 'AND' network.
    # All nodes are off.
    s.subsys_all_off = Subsystem(s.nodes,
                                 np.array([0, 0, 0]),
                                 np.array([0, 0, 0]),
                                 s)
    return s


def big():
    """ Large network.

    TPM:


    """
    tpm = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]]).reshape([2] * 5 + [5],
                                              order="F").astype(float)
    # All on
    current_state = np.array([1] * 5)
    # All on
    past_state = np.array([1] * 5)
    big = Network(tpm, current_state, past_state)
    big.subsys_all = Subsystem(big.nodes, current_state, past_state, big)
    return big

def reducible():
    tpm = np.zeros([2] * 2 + [2])
    current_state = np.zeros(2)
    past_state = np.zeros(2)
    cm = np.array([[1, 0],
                   [0, 1]])
    r = Network(tpm, current_state, past_state, connectivity_matrix=cm)
    # Return the full subsystem
    return Subsystem(r.nodes, current_state, past_state, r)
