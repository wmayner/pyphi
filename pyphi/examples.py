#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# examples.py
"""
Example networks and subsystems to go along with the documentation.
"""

import numpy as np
from .network import Network
from .subsystem import Subsystem


def basic_network():
    """A simple 3-node network with roughly two bits of |big_phi|.

    Diagram::

               +~~~~~~~~+
          +~~~~|   A    |<~~~~+
          |    |  (OR)  +~~~+ |
          |    +~~~~~~~~+   | |
          |                 | |
          |                 v |
        +~+~~~~~~+      +~~~~~+~+
        |   B    |<~~~~~+   C   |
        | (COPY) +~~~~~>| (XOR) |
        +~~~~~~~~+      +~~~~~~~+

    TPM:

    +--------------+---------------+
    |  Past state  | Current state |
    +--------------+---------------+
    |   A, B, C    |    A, B, C    |
    +==============+===============+
    |   0, 0, 0    |    0, 0, 0    |
    +--------------+---------------+
    |   1, 0, 0    |    0, 0, 1    |
    +--------------+---------------+
    |   0, 1, 0    |    1, 0, 1    |
    +--------------+---------------+
    |   1, 1, 0    |    1, 0, 0    |
    +--------------+---------------+
    |   0, 0, 1    |    1, 1, 0    |
    +--------------+---------------+
    |   1, 0, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   0, 1, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   1, 1, 1    |    1, 1, 0    |
    +--------------+---------------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 0 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+

    .. note::

        |CM[i][j] = 1| means that node |i| is connected to node |j|.
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ])

    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (1, 0, 0)
    past_state = (1, 1, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def basic_subsystem():
    """A subsystem containing all the nodes of the
    :func:`pyphi.examples.basic_network`."""
    net = basic_network()
    return Subsystem(range(net.size), net)


def residue_network():
    """The network for the residue example.

    Current and past state are all nodes off.

    Diagram::

                +~~~~~~~+         +~~~~~~~+
                |   A   |         |   B   |
            +~~>| (AND) |         | (AND) |<~~+
            |   +~~~~~~~+         +~~~~~~~+   |
            |        ^               ^        |
            |        |               |        |
            |        +~~~~~+   +~~~~~+        |
            |              |   |              |
        +~~~+~~~+        +~+~~~+~+        +~~~+~~~+
        |   C   |        |   D   |        |   E   |
        |       |        |       |        |       |
        +~~~~~~~+        +~~~~~~~+        +~~~~~~~+

    Connectivity matrix:

    +---+---+---+---+---+---+
    | . | A | B | C | D | E |
    +---+---+---+---+---+---+
    | A | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | B | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | C | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | D | 1 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | E | 0 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    """
    tpm = np.array([
        [int(s) for s in bin(x)[2:].zfill(5)[::-1]] for x in range(32)
    ])
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) == 2), 0] = 1
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) == 2), 1] = 1
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) < 2), 0] = 0
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) < 2), 1] = 0

    cm = np.zeros((5, 5))
    cm[2:4, 0] = 1
    cm[3:, 1] = 1

    current_state = (0, 0, 0, 0, 0)
    past_state = (0, 0, 0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def residue_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.residue_network`."""
    net = residue_network()
    return Subsystem(range(net.size), net)


def xor_network():
    """A fully connected system of three XOR gates. In the state ``(0, 0, 0)``,
    none of the elementary mechanisms exist.

    Diagram::

        +~~~~~~~+       +~~~~~~~+
        |   A   +<~~~~~>|   B   |
        | (XOR) |       | (XOR) |
        +~~~~~~~+       +~~~~~~~+
            ^               ^
            |   +~~~~~~~+   |
            +~~>|   C   |<~~+
                | (XOR) |
                +~~~~~~~+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])

    cm = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (0, 0, 0)
    past_state = (0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def xor_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.xor_network`."""
    net = xor_network()
    return Subsystem(range(net.size), net)


def cond_depend_tpm():
    """A system of two general logic gates A and B such if they are in the same
    state they stay the same, but if they are in different states, they flip
    with probability 50%.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+

    TPM:

    +------+------+------+------+------+
    |      |(0, 0)|(1, 0)|(0, 1)|(1, 1)|
    +------+------+------+------+------+
    |(0, 0)| 1.0  | 0.0  | 0.0  | 0.0  |
    +------+------+------+------+------+
    |(1, 0)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(0, 1)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(1, 1)| 0.0  | 0.0  | 0.0  | 1.0  |
    +------+------+------+------+------+

    Connectivity matrix:

    +---+---+---+
    | . | A | B |
    +---+---+---+
    | A | 0 | 1 |
    +---+---+---+
    | B | 1 | 0 |
    +---+---+---+
    """

    tpm = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return tpm


def cond_independ_tpm():
    """A system of three general logic gates A, B and C such that if A and B
    are in the same state then they stay the same. If they are in different
    states, they flip if C is ''ON and stay the same if C is OFF. Node C is ON
    50% of the time, independent of the previous state.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+
           ^               ^
           |    +~~~~~+    |
           +~~~~+  C  +~~~~+
                +~~~~~+

    TPM:

    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |         |(0, 0, 0)|(1, 0, 0)|(0, 1, 0)|(1, 1, 0)|(0, 0, 1)|(1, 0, 1)|(0, 1, 1)|(1, 1, 1)|
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 0)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 0)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 0)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 0)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 1)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 1)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 1)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 1)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 0 |
    +---+---+---+---+
    | B | 1 | 0 | 0 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """

    tpm = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
    ])

    return tpm
