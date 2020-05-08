#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# examples.py

"""
Example networks and subsystems to go along with the documentation.
"""

# pylint: disable=too-many-lines
# flake8: noqa

import string

import numpy as np

from .actual import Transition
from .network import Network
from .subsystem import Subsystem
from .utils import all_states

LABELS = string.ascii_uppercase


# TODO(relations): add docstring
def PQR_network():
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ])
    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=['P', 'Q', 'R'])


# TODO(relations): add docstring
def PQR():
    return Subsystem(PQR_network(), (1, 0, 0))


def basic_network(cm=False):
    """A 3-node network of logic gates.

    Diagram::

                +~~~~~~~~+
          +~~~~>|   A    |<~~~~+
          |     |  (OR)  +~~~+ |
          |     +~~~~~~~~+   | |
          |                  | |
          |                  v |
        +~+~~~~~~+       +~~~~~+~+
        |   B    |<~~~~~~+   C   |
        | (COPY) +~~~~~~>| (XOR) |
        +~~~~~~~~+       +~~~~~~~+

    TPM:

    +----------------+---------------+
    | Previous state | Current state |
    +----------------+---------------+
    |    A, B, C     |    A, B, C    |
    +================+===============+
    |    0, 0, 0     |    0, 0, 0    |
    +----------------+---------------+
    |    1, 0, 0     |    0, 0, 1    |
    +----------------+---------------+
    |    0, 1, 0     |    1, 0, 1    |
    +----------------+---------------+
    |    1, 1, 0     |    1, 0, 0    |
    +----------------+---------------+
    |    0, 0, 1     |    1, 1, 0    |
    +----------------+---------------+
    |    1, 0, 1     |    1, 1, 1    |
    +----------------+---------------+
    |    0, 1, 1     |    1, 1, 1    |
    +----------------+---------------+
    |    1, 1, 1     |    1, 1, 0    |
    +----------------+---------------+

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
        |CM[i][j] = 1| means that there is a directed edge |(i,j)| from node
        |i| to node |j| and |CM[i][j] = 0| means there is no edge from |i| to
        |j|.
    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ])
    if cm is False:
        cm = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
    # fmt: on
    else:
        cm = None
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def basic_state():
    """The state of nodes in :func:`~pyphi.examples.basic_network`."""
    return (1, 0, 0)


def basic_subsystem():
    """A subsystem containing all the nodes of the
    :func:`~pyphi.examples.basic_network`.
    """
    net = basic_network()
    state = basic_state()
    return Subsystem(net, state)


def basic_noisy_selfloop_network():
    """Based on the basic_network, but with added selfloops and noisy edges.

    Nodes perform deterministic functions of their inputs, but those inputs
    may be flipped (i.e. what should be a 0 becomes a 1, and vice versa) with
    probability epsilon (eps = 0.1 here).

    Diagram::

                   +~~+
                   |  v
                +~~~~~~~~+
          +~~~~>|   A    |<~~~~+
          |     |  (OR)  +~~~+ |
          |     +~~~~~~~~+   | |
          |                  | |
          |                  v |
        +~+~~~~~~+       +~~~~~+~+
        |   B    |<~~~~~~+   C   |
      +>| (COPY) +~~~~~~>| (XOR) |<+
      | +~~~~~~~~+       +~~~~~~~+ |
      |   |                    |   |
      +~~~+                    +~~~+

    """
    # fmt: off
    tpm = np.array([
        [0.271, 0.19, 0.244],
        [0.919, 0.19, 0.756],
        [0.919, 0.91, 0.756],
        [0.991, 0.91, 0.244],
        [0.919, 0.91, 0.756],
        [0.991, 0.91, 0.244],
        [0.991, 0.99, 0.244],
        [0.999, 0.99, 0.756],
    ])
    cm = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    # fmt: on
    return Network(tpm, cm=cm)


def basic_noisy_selfloop_subsystem():
    """A subsystem containing all the nodes of the
    :func:`~pyphi.examples.basic_noisy_selfloop_network`.
    """
    net = basic_noisy_selfloop_network()
    state = basic_state()
    return Subsystem(net, state)


def residue_network():
    """The network for the residue example.

    Current and previous state are all nodes OFF.

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

    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def residue_subsystem():
    """The subsystem containing all the nodes of the
    :func:`~pyphi.examples.residue_network`.
    """
    net = residue_network()
    state = (0, 0, 0, 0, 0)
    return Subsystem(net, state)


def xor_network():
    """A fully connected system of three XOR gates. In the state ``(0, 0, 0)``,
    none of the elementary mechanisms exist.

    Diagram::

        +~~~~~~~+       +~~~~~~~+
        |   A   +<~~~~~~+   B   |
        | (XOR) +~~~~~~>| (XOR) |
        +~+~~~~~+       +~~~~~+~+
          | ^               ^ |
          | |   +~~~~~~~+   | |
          | +~~~+   C   +~~~+ |
          +~~~~>| (XOR) +<~~~~+
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
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ])
    cm = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def xor_subsystem():
    """The subsystem containing all the nodes of the
    :func:`~pyphi.examples.xor_network`.
    """
    net = xor_network()
    state = (0, 0, 0)
    return Subsystem(net, state)


def cond_depend_tpm():
    """A system of two general logic gates A and B such if they are in the same
    state they stay the same, but if they are in different states, they flip
    with probability 50%.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~~+  B  |
        |     +~~~~~~~~>|     |
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
    # fmt: off
    tpm = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # fmt: on
    return tpm


def cond_independ_tpm():
    """A system of three general logic gates A, B and C such that: if A and B
    are in the same state then they stay the same; if they are in different
    states, they flip if C is ON and stay the same if C is OFF; and C is ON 50%
    of the time, independent of the previous state.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  +~~~~~~~~>|  B  |
        |     |<~~~~~~~~+     |
        +~+~~~+         +~~~+~+
          | ^             ^ |
          | |   +~~~~~+   | |
          | ~~~~+  C  +~~~+ |
          +~~~~>|     |<~~~~+
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
    # fmt: off
    tpm = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
    ])
    # fmt: on
    return tpm


def propagation_delay_network():
    """A version of the primary example from the IIT 3.0 paper with
    deterministic COPY gates on each connection. These copy gates essentially
    function as propagation delays on the signal between OR, AND and XOR gates
    from the original system.

    The current and previous states of the network are also selected to mimic
    the corresponding states from the IIT 3.0 paper.

    Diagram::

                                   +----------+
                +------------------+ C (COPY) +<----------------+
                v                  +----------+                 |
        +-------+-+                                           +-+-------+
        |         |                +----------+               |         |
        | A (OR)  +--------------->+ B (COPY) +-------------->+ D (XOR) |
        |         |                +----------+               |         |
        +-+-----+-+                                           +-+-----+-+
          |     ^                                               ^     |
          |     |                                               |     |
          |     |   +----------+                 +----------+   |     |
          |     +---+ H (COPY) +<----+     +---->+ F (COPY) +---+     |
          |         +----------+     |     |     +----------+         |
          |                          |     |                          |
          |                        +-+-----+-+                        |
          |         +----------+   |         |   +----------+         |
          +-------->+ I (COPY) +-->| G (AND) |<--+ E (COPY) +<--------+
                    +----------+   |         |   +----------+
                                   +---------+

    Connectivity matrix:

    +---+---+---+---+---+---+---+---+---+---+
    | . | A | B | C | D | E | F | G | H | I |
    +---+---+---+---+---+---+---+---+---+---+
    | A | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
    +---+---+---+---+---+---+---+---+---+---+
    | B | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | C | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | D | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | E | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | F | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | G | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | H | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | I | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+

    States:

    In the IIT 3.0 paper example, the previous state of the system has only the
    XOR gate ON. For the propagation delay network, this corresponds to a state
    of
    ``(0, 0, 0, 1, 0, 0, 0, 0, 0)``.

    The current state of the IIT 3.0 example has only the OR gate ON. By
    advancing the propagation delay system two time steps, the current state
    ``(1, 0, 0, 0, 0, 0, 0, 0, 0)`` is achieved, with corresponding previous
    state ``(0, 0, 1, 0, 1, 0, 0, 0, 0)``.
    """
    num_nodes = 9
    num_states = 2 ** num_nodes

    tpm = np.zeros((num_states, num_nodes))

    for previous_state_index, previous in enumerate(all_states(num_nodes)):
        current_state = [0 for i in range(num_nodes)]
        if previous[2] == 1 or previous[7] == 1:
            current_state[0] = 1
        if previous[0] == 1:
            current_state[1] = 1
            current_state[8] = 1
        if previous[3] == 1:
            current_state[2] = 1
            current_state[4] = 1
        if previous[1] == 1 ^ previous[5] == 1:
            current_state[3] = 1
        if previous[4] == 1 and previous[8] == 1:
            current_state[6] = 1
        if previous[6] == 1:
            current_state[5] = 1
            current_state[7] = 1
        tpm[previous_state_index, :] = current_state

    # fmt: off
    cm = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
    ])
    # fmt: on

    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def macro_network():
    """A network of micro elements which has greater integrated information
    after coarse graining to a macro scale.
    """
    # fmt: off
    tpm = np.array([
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 1.0, 1.0],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 1.0, 1.0],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 1.0, 1.0],
        [1.0, 1.0, 0.3, 0.3],
        [1.0, 1.0, 0.3, 0.3],
        [1.0, 1.0, 0.3, 0.3],
        [1.0, 1.0, 1.0, 1.0],
    ])
    # fmt: on
    return Network(tpm, node_labels=LABELS[:tpm.shape[1]])


def macro_subsystem():
    """A subsystem containing all the nodes of
    :func:`~pyphi.examples.macro_network`.
    """
    net = macro_network()
    state = (0, 0, 0, 0)
    return Subsystem(net, state)


def blackbox_network():
    """A micro-network to demonstrate blackboxing.

    Diagram::

                                +----------+
          +-------------------->+ A (COPY) + <---------------+
          |                     +----------+                 |
          |                 +----------+                     |
          |     +-----------+ B (COPY) + <-------------+     |
          v     v           +----------+               |     |
        +-+-----+-+                                  +-+-----+-+
        |         |                                  |         |
        | C (AND) |                                  | F (AND) |
        |         |                                  |         |
        +-+-----+-+                                  +-+-----+-+
          |     |                                      ^     ^
          |     |           +----------+               |     |
          |     +---------> + D (COPY) +---------------+     |
          |                 +----------+                     |
          |                     +----------+                 |
          +-------------------> + E (COPY) +-----------------+
                                +----------+

    Connectivity Matrix:

    +---+---+---+---+---+---+---+
    | . | A | B | C | D | E | F |
    +---+---+---+---+---+---+---+
    | A | 0 | 0 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+
    | B | 0 | 0 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+
    | C | 0 | 0 | 0 | 1 | 1 | 0 |
    +---+---+---+---+---+---+---+
    | D | 0 | 0 | 0 | 0 | 0 | 1 |
    +---+---+---+---+---+---+---+
    | E | 0 | 0 | 0 | 0 | 0 | 1 |
    +---+---+---+---+---+---+---+
    | F | 1 | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+


    In the documentation example, the state is (0, 0, 0, 0, 0, 0).
    """
    num_nodes = 6
    num_states = 2 ** num_nodes
    tpm = np.zeros((num_states, num_nodes))

    for index, previous_state in enumerate(all_states(num_nodes)):
        current_state = [0 for i in range(num_nodes)]
        if previous_state[5] == 1:
            current_state[0] = 1
            current_state[1] = 1
        if previous_state[0] == 1 and previous_state[1]:
            current_state[2] = 1
        if previous_state[2] == 1:
            current_state[3] = 1
            current_state[4] = 1
        if previous_state[3] == 1 and previous_state[4] == 1:
            current_state[5] = 1
        tpm[index, :] = current_state

    # fmt: off
    cm = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
    ])
    # fmt: on

    return Network(tpm, cm, node_labels=LABELS[:tpm.shape[1]])


def rule110_network():
    """A network of three elements which follows the logic of the Rule 110
    cellular automaton with current and previous state (0, 0, 0).
    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    ])
    # fmt: on
    return Network(tpm, node_labels=LABELS[:tpm.shape[1]])


def rule154_network():
    """A network of three elements which follows the logic of the Rule 154
    cellular automaton.
    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ])
    cm = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig1a():
    """The network shown in Figure 1A of the 2014 IIT 3.0 paper."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
    ])
    cm = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig3a():
    """The network shown in Figure 3A of the 2014 IIT 3.0 paper."""
    # fmt: off
    tpm = np.array([
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0.5, 0, 0, 0],
    ])
    cm = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig3b():
    """The network shown in Figure 3B of the 2014 IIT 3.0 paper."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    cm = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig4():
    """The network shown in Figures 4, 6, 8, 9 and 10 of the 2014 IIT 3.0 paper.

    Diagram::

                +~~~~~~~+
          +~~~~>|   A   |<~~~~+
          | +~~~+ (OR)  +~~~+ |
          | |   +~~~~~~~+   | |
          | |               | |
          | v               v |
        +~+~~~~~+       +~~~~~+~+
        |   B   |<~~~~~~+   C   |
        | (AND) +~~~~~~>| (XOR) |
        +~~~~~~~+       +~~~~~~~+

    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    cm = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig5a():
    """The network shown in Figure 5A of the 2014 IIT 3.0 paper.

    Diagram::

                 +~~~~~~~+
           +~~~~>|   A   |<~~~~+
           |     | (AND) |     |
           |     +~~~~~~~+     |
           |                   |
        +~~+~~~~~+       +~~~~~+~~+
        |    B   |<~~~~~~+   C    |
        | (COPY) +~~~~~~>| (COPY) |
        +~~~~~~~~+       +~~~~~~~~+

    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])
    cm = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


def fig5b():
    """The network shown in Figure 5B of the 2014 IIT 3.0 paper.

    Diagram::

                 +~~~~~~~+
            +~~~~+   A   +~~~~+
            |    | (AND) |    |
            |    +~~~~~~~+    |
            v                 v
        +~~~~~~~~+       +~~~~~~~~+
        |    B   |<~~~~~~+   C    |
        | (COPY) +~~~~~~>| (COPY) |
        +~~~~~~~~+       +~~~~~~~~+

    """
    # fmt: off
    tpm = np.array([
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    cm = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


# The networks in figures 4, 6 and 8 are the same.
fig6, fig8, fig9, fig10 = 4 * (fig4,)

# The network in Figure 14 is the same as that in Figure 1A.
fig14 = fig1a


def fig16():
    """The network shown in Figure 5B of the 2014 IIT 3.0 paper."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
    ])
    cm = np.array([
        [0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ])
    # fmt: on
    return Network(tpm, cm=cm, node_labels=LABELS[:tpm.shape[1]])


# Actual Causation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def actual_causation():
    """The actual causation example network, consisting of an ``OR`` and
    ``AND`` gate with self-loops.
    """
    # fmt: off
    tpm = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    cm = np.array([
        [1, 1],
        [1, 1],
    ])
    # fmt: on
    return Network(tpm, cm, node_labels=('OR', 'AND'))


def disjunction_conjunction_network():
    """The disjunction-conjunction example from Actual Causation Figure 7.

    A network of four elements, one output ``D`` with three inputs ``A B C``.
    The output turns ON if ``A`` AND ``B`` are ON or if ``C`` is ON.
    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    cm = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    # fmt: on
    return Network(tpm, cm, node_labels=LABELS[:tpm.shape[1]])


def prevention():
    """The |Transition| for the prevention example from Actual Causation
    Figure 5D.
    """
    # fmt: off
    tpm = np.array([
        [0.5, 0.5, 1],
        [0.5, 0.5, 0],
        [0.5, 0.5, 1],
        [0.5, 0.5, 1],
        [0.5, 0.5, 1],
        [0.5, 0.5, 0],
        [0.5, 0.5, 1],
        [0.5, 0.5, 1],
    ])
    cm = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])
    # fmt: on
    network = Network(tpm, cm, node_labels=['A', 'B', 'F'])
    x_state = (1, 1, 1)
    y_state = (1, 1, 1)

    return Transition(network, x_state, y_state, (0, 1), (2,))
