# examples.py
"""Example substrates and systems to go along with the documentation."""

# pylint: disable=too-many-lines
# flake8: noqa

import string
from collections import defaultdict
from dataclasses import replace

import numpy as np

from . import actual
from .actual import Transition
from .conf import config
from .substrate import Substrate
from .substrate_generator import build_substrate, ising
from .system import System
from .utils import all_states, powerset

LABELS = string.ascii_uppercase

EXAMPLES = defaultdict(dict)


def register_example(func):
    name = func.__name__.split("_")
    obj = name[-1]
    name = "_".join(name[:-1])
    EXAMPLES[obj][name] = func
    return func


@register_example
def grid3_substrate():
    """3-node grid substrate."""
    # Grid
    # fmt: off
    tpm = np.array([
       [[[0.04742587, 0.02931223, 0.04742587],
         [0.04742587, 0.07585818, 0.88079708]],

        [[0.11920292, 0.81757448, 0.11920292],
         [0.11920292, 0.92414182, 0.95257413]]],


       [[[0.88079708, 0.07585818, 0.04742587],
         [0.88079708, 0.18242552, 0.88079708]],

        [[0.95257413, 0.92414182, 0.11920292],
         [0.95257413, 0.97068777, 0.95257413]]]
    ])
    cm = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ])
    # fmt: on
    return Substrate(tpm, cm=cm, node_labels=["A", "B", "C"])


@register_example
def grid3_system():
    return System(grid3_substrate(), state=(0, 0, 0), node_indices=(0, 1, 2))


@register_example
def basic_substrate(cm=False):
    """A 3-node substrate of logic gates.

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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


def basic_state():
    """The state of nodes in :func:`~pyphi.examples.basic_substrate`."""
    return (1, 0, 0)


@register_example
def basic_system():
    """A system containing all the nodes of the
    :func:`~pyphi.examples.basic_substrate`.
    """
    net = basic_substrate()
    state = basic_state()
    return System(net, state)


# TODO(relations): add docstring
pqr_substrate = basic_substrate
pqr_system = basic_system


@register_example
def basic_noisy_selfloop_substrate():
    """Based on the basic_substrate, but with added selfloops and noisy edges.

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
    return Substrate(tpm, cm=cm)


@register_example
def basic_noisy_selfloop_system():
    """A system containing all the nodes of the
    :func:`~pyphi.examples.basic_noisy_selfloop_substrate`.
    """
    net = basic_noisy_selfloop_substrate()
    state = basic_state()
    return System(net, state)


@register_example
@config.override(validate_connectivity=False)
def residue_substrate():
    """The substrate for the residue example.

    The input units C, D, E carry self-copying TPM columns, but the
    connectivity matrix (matching the diagram below) gives them no inputs, so
    their self-state is marginalized to an unconstrained (uniform) effect
    repertoire — the intended input-unit behavior. The connectivity-consistency
    check is disabled (via the decorator) because it would otherwise flag those
    self-loops.

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
    tpm = np.array([[int(s) for s in bin(x)[2:].zfill(5)[::-1]] for x in range(32)])
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) == 2), 0] = 1
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) == 2), 1] = 1
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) < 2), 0] = 0
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) < 2), 1] = 0

    cm = np.zeros((5, 5))
    cm[2:4, 0] = 1
    cm[3:, 1] = 1

    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def residue_system():
    """The system containing all the nodes of the
    :func:`~pyphi.examples.residue_substrate`.
    """
    net = residue_substrate()
    state = (0, 0, 0, 0, 0)
    return System(net, state)


@register_example
def xor_substrate():
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def xor_system():
    """The system containing all the nodes of the
    :func:`~pyphi.examples.xor_substrate`.
    """
    net = xor_substrate()
    state = (0, 0, 0)
    return System(net, state)


@register_example
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


@register_example
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


@register_example
def propagation_delay_substrate():
    """A version of the primary example from the IIT 3.0 paper with
    deterministic COPY gates on each connection. These copy gates essentially
    function as propagation delays on the signal between OR, AND and XOR gates
    from the original system.

    The current and previous states of the substrate are also selected to mimic
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
    XOR gate ON. For the propagation delay substrate, this corresponds to a state
    of
    ``(0, 0, 0, 1, 0, 0, 0, 0, 0)``.

    The current state of the IIT 3.0 example has only the OR gate ON. By
    advancing the propagation delay system two time steps, the current state
    ``(1, 0, 0, 0, 0, 0, 0, 0, 0)`` is achieved, with corresponding previous
    state ``(0, 0, 1, 0, 1, 0, 0, 0, 0)``.
    """
    num_nodes = 9
    num_states = 2**num_nodes

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

    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def macro_substrate():
    """A substrate of micro elements which has greater integrated information
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
    return Substrate(tpm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def macro_system():
    """A system containing all the nodes of
    :func:`~pyphi.examples.macro_substrate`.
    """
    net = macro_substrate()
    state = (0, 0, 0, 0)
    return System(net, state)


@register_example
def blackbox_substrate():
    """A micro-substrate to demonstrate blackboxing.

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
    num_states = 2**num_nodes
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

    return Substrate(tpm, cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def rule110_substrate():
    """A substrate of three elements which follows the logic of the Rule 110
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
        [0, 0, 0]
    ])
    # fmt: on
    return Substrate(tpm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def rule110_system():
    return System(rule110_substrate(), (0, 0, 0), node_indices=(0, 1, 2))


@register_example
def rule154_substrate():
    """A substrate of five elements which follows the logic of the Rule 154
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def rule154_system():
    return System(rule154_substrate(), (0,) * 5)


@register_example
def fig1a_substrate():
    """The substrate shown in Figure 1A of the 2014 IIT 3.0 paper."""
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig3a_substrate():
    """The substrate shown in Figure 3A of the 2014 IIT 3.0 paper."""
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig3b_substrate():
    """The substrate shown in Figure 3B of the 2014 IIT 3.0 paper."""
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig4_substrate():
    """The substrate shown in Figures 4, 6, 8, 9 and 10 of the 2014 IIT 3.0 paper.

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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig4_system():
    return System(fig4_substrate(), state=(1, 0, 1), node_indices=(0, 1, 2))


@register_example
def fig5a_substrate():
    """The substrate shown in Figure 5A of the 2014 IIT 3.0 paper.

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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig5a_system():
    return System(fig5a_substrate(), state=(0, 0, 0), node_indices=(0, 1, 2))


@register_example
def fig5b_substrate():
    """The substrate shown in Figure 5B of the 2014 IIT 3.0 paper.

    Diagram::

                 +~~~~~~~+
            +~~~~+   A   +~~~~+
            |    |       |    |
            |    +~~~~~~~+    |
            v                 v
        +~~~~~~~~+       +~~~~~~~~+
        |    B   |<~~~~~~+   C    |
        | (AND)  +~~~~~~>|  (OR)  |
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def fig5b_system():
    return System(fig5b_substrate(), state=(1, 0, 1), node_indices=(0, 1, 2))


# The substrates in figures 4, 6 and 8 are the same.
fig6_substrate = fig8_substrate = fig9_substrate = fig10_substrate = fig4_substrate

# The substrate in Figure 14 is the same as that in Figure 1A.
fig14_substrate = fig1a_substrate


@register_example
def fig16_substrate():
    """The substrate shown in Figure 5B of the 2014 IIT 3.0 paper."""
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
    return Substrate(tpm, cm=cm, node_labels=LABELS[: tpm.shape[1]])


# Actual Causation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@register_example
def actual_causation_substrate():
    """The actual causation example substrate, consisting of an ``OR`` and
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
    return Substrate(tpm, cm, node_labels=("OR", "AND"))


@register_example
def disjunction_conjunction_substrate():
    """The disjunction-conjunction example from Actual Causation Figure 7.

    A substrate of four elements, one output ``D`` with three inputs ``A B C``.
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
    return Substrate(tpm, cm, node_labels=LABELS[: tpm.shape[1]])


@register_example
def gomez_p53_mdm2_substrate():
    """The multi-valued p53-Mdm2 regulatory network from Gómez et al. (2020).

    A three-element *non-binary* substrate from "PyPhi for multi-valued
    elements" (Gómez, Mayner, Beheler-Amass, Tononi & Albantakis, *Entropy*
    23(1):6; network in Fig. 3A, evolution function in Table 3), itself the
    logical model of Abou-Jaoudé et al. The units are the tumour-suppressor
    ``P`` (p53), modelled as *ternary* (activity levels 0/1/2), and the nuclear
    and cytoplasmic forms of the ubiquitin ligase Mdm2, ``Mn`` and ``Mc`` (each
    *binary*). Node (and state) order is ``(P, Mc, Mn)`` and the deterministic
    evolution function from Table 3 is::

        P'  = 2  if Mn == 0 else 0          (nuclear Mdm2 degrades p53)
        Mc' = 1  if P == 2  else 0          (high p53 up-regulates cyto. Mdm2)
        Mn' = 0  if (Mc == 0 and P >= 1) else 1

    The fixed point analysed in the paper (Fig. 3A) is ``(P, Mc, Mn) = (0, 0,
    1)``.
    """
    alphabet = (3, 2, 2)  # P ternary; Mc, Mn binary
    update = (
        lambda p, mc, mn: 2 if mn == 0 else 0,
        lambda p, mc, mn: 1 if p == 2 else 0,
        lambda p, mc, mn: 0 if (mc == 0 and p >= 1) else 1,
    )
    marginals = []
    for i, k in enumerate(alphabet):
        factor = np.zeros((*alphabet, k))
        for state in np.ndindex(*alphabet):
            factor[(*state, update[i](*state))] = 1.0
        marginals.append(factor)
    return Substrate(
        marginals=marginals,
        state_space=((0, 1, 2), (0, 1), (0, 1)),
        node_labels=("P", "Mc", "Mn"),
    )


@register_example
def prevention_transition():
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
    substrate = Substrate(tpm, cm, node_labels=["A", "B", "F"])
    x_state = (1, 1, 1)
    y_state = (1, 1, 1)

    return Transition(substrate, x_state, y_state, (0, 1), (2,))


@register_example
@config.override(
    iit=replace(
        config.formalism.iit,
        mechanism_partition_scheme="WEDGE_TRIPARTITION",
        mechanism_phi_measure="BLD",
    ),
    validate_system_states=False,
    alpha_measure="WPMI",
)
def frog_example():
    """
    Example used in the paper::

        Causal reductionism and causal structures
        Grasso, M, Albantakis, L, Lang, J, & Tononi, G

    """

    def LogFunc(x, l, k, x0):
        y = 1 / (l + np.e ** (-k * (x - x0)))
        return y

    def Gauss(x, mu, si):
        y = np.exp(-0.5 * (((x - mu) / si) ** 2))
        return y

    def NR(x, exponent, threshold):
        x_exp = x**exponent
        y = x_exp / (threshold + x_exp)
        return y

    def get_net(
        mech_func,
        weights,
        mu=None,
        si=None,
        exp=None,
        th=None,
        l=None,
        k=None,
        x0=None,
        input_nodes=None,
        input_modifier=None,
        node_labels=None,
        substrate_name=None,
        pickle_substrate=True,
    ):
        """
        Returns a pyphi substrate (with the specified activation function)

        Args:
            mech_func: (list) list of mechanism function labels ('g' for Gaussian, 'nr' or 's' for Naka-Rushton, 'l' for LogFunc)
            weights: (numpy array) matrix of node by node weights (x sends to y)
            mu = mean (Gauss)
            si = standard deviation (Gauss)
            exp = exponent (NR or MvsG)
            th = threshold (NR) or curve steepness (MvsG)
            x0 = midpoint value (LogFunc)
            l = max value (LogFunc)
            k = growth rate (LogFunc)
            gridsize = number of substrate nodes in the grid excluded inputs
        """
        weights = weights.T
        node_indices = [n for n in range(len(weights))]
        nodes_n = len(node_indices)

        if node_labels is None:
            node_labels = [string.ascii_uppercase[n] for n in range(len(weights))]

        mechs_pset = list(powerset(range(nodes_n), nonempty=True))
        states = list(all_states(nodes_n))
        tpm = np.zeros([2**nodes_n, nodes_n])

        for s in range(len(states)):
            state = states[s]
            tpm_line = []

            for z in node_indices:
                # g = Gaussian
                if mech_func[z] == "g":
                    val = Gauss(
                        sum(state * np.array([weights[z][n] for n in node_indices])),
                        mu,
                        si,
                    )
                # nr = Naka Rushton, s = space
                elif mech_func[z] == "nr" or mech_func[z] == "s":
                    input_sum = sum(state * weights[z])
                    val = NR(input_sum, exp, th)
                # l = LogFunc
                elif mech_func[z] == "l":
                    val = LogFunc(
                        sum(state * np.array([weights[z][n] for n in node_indices])),
                        l,
                        k,
                        x0,
                    )
                # i = inhibiting input
                elif mech_func[z] == "i":
                    assert input_nodes is not None, (
                        "input_nodes required for inhibiting input"
                    )
                    assert input_modifier is not None, (
                        "input_modifier required for inhibiting input"
                    )
                    non_input_nodes = [n for n in node_indices if n not in input_nodes]
                    input_weights = [
                        -input_modifier if state[n] == 0 else 1 for n in input_nodes
                    ] * np.array([weights[z][n] for n in input_nodes])
                    other_weights = [state[n] for n in non_input_nodes] * np.array(
                        [weights[z][n] for n in non_input_nodes]
                    )
                    weights_sum = sum(input_weights) + sum(other_weights)
                    val = Gauss(weights_sum, mu, si)
                else:
                    raise NameError("Mechanism function not recognized")

                tpm_line.append(val)

            tpm[s] = tuple(tpm_line)

        cm = np.array(
            [[1.0 if w else 0 for w in weights[n]] for n in range(len(weights))]
        )
        cm = cm.T
        substrate = Substrate(tpm, cm, node_labels)

        return substrate

    # F3 Frog
    print("F3 frog:\n")
    mu = 1
    si = 0.3

    mech_func = ["g", "g", "g", "g", "g", "g", "g", "g"]
    #'S1','S2','S3','H1','H2','H3','M1','M2'

    weights = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],  # S1
            [0.0, 0.0, 0.0, 0.5, 0.9, 0.5, 0.0, 0.0],  # S2
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],  # S3
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],  # H1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2],  # H3
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # H2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M2
        ]
    )  # S1,S2,S3,H1,H3,H2,M1,M2

    node_labels = ["SL", "SC", "SR", "CL", "CC", "CR", "ML", "MR"]

    substrate = get_net(mech_func, weights, mu=mu, si=si, node_labels=node_labels)

    transition = actual.Transition(
        substrate,
        (1, 0, 1, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1, 1, 1),
        (0, 1, 2, 3, 4, 5),
        (3, 4, 5, 6, 7),
    )
    print(transition)
    account = actual.account(transition)
    print(account)

    # F2 Frog
    print("F2 frog:\n")

    mu = 1
    si = 0.3

    mech_func = [
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
    ]
    #'S1','S2','S3', N1','N2','M1','M2',

    weights = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # S1
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],  # S2
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # S3
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],  # H1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8],  # H2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M2
        ]
    )  # S1,S2,S3,H1,H2,M1,M2

    node_labels = ["SL", "SC", "SR", "CL", "CR", "ML", "MR"]

    substrate = get_net(mech_func, weights, mu=mu, si=si, node_labels=node_labels)

    transition = actual.Transition(
        substrate,
        (1, 0, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1, 1),
        (0, 1, 2, 3, 4),
        (3, 4, 5, 6),
    )
    print(transition)
    account = actual.account(transition)
    print(account)

    # F1 Frog
    print("\n\nF1 frog:\n")
    mu = 1
    si = 0.3

    mech_func = ["g", "g", "g", "g", "g", "g", "g", "g"]
    #'S1','S2','S3','S4','N1','N2','M1','M2',

    weights = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # S1
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # S2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],  # S3
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # S4
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # H1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # H2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # M2
        ]
    )  # S1,S2,S3,S4,H1,H2,M1,M2

    node_labels = ["S1", "S2", "S3", "S4", "H1", "H2", "M1", "M2"]

    substrate = get_net(mech_func, weights, mu=mu, si=si, node_labels=node_labels)

    transition = actual.Transition(
        substrate,
        (1, 0, 0, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 1, 1, 1),
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7),
    )
    print(transition)
    account = actual.account(transition)
    print(account)


def differentiation_micro_tpm(p, epsilon):
    # Two noisy AND gates connected to each other
    return np.minimum(
        1, np.array([[p, p], [p, p + epsilon], [p + epsilon, p], [1 - p, 1 - p]])
    )


def differentiation_macro_tpm(p, epsilon):
    # Two noisy AND gates connected to each other, macroed into two states:
    #   (1, 1) -> 1
    #   all other states -> 0
    return np.minimum(
        1, np.array([[(p * p + 2 * p * epsilon) / 3], [(1 - p) * (1 - p)]])
    )


@register_example
def differentiation_micro_1_system():
    return System(
        substrate=Substrate(differentiation_micro_tpm(0.9, 0.01)),
        state=(0, 0),
        node_indices=(0, 1),
    )


# --------------------------------------------------------------------------- #
# IIT 4.0 (2023) -- Albantakis et al., PLoS Comput Biol 19(10): e1011465
# --------------------------------------------------------------------------- #
# The example architectures of Figures 6 and 7 are Ising networks: each unit's
# next state is a logistic (sigmoid) function of its weighted inputs in {-1, +1}
# with slope k = 4, realized by `ising.probability` at `temperature = 1 / k`.
# The connection-weight matrices below are the exact definitions provided by the
# paper's authors (weights[i, j] is the weight of the edge from unit i to unit j).


@register_example
def iit4_2023_fig6a_substrate():
    """The 6-unit "bottleneck" architecture of Fig 6A.

    Unit A both drives and is driven by a fan of five units (B-F), each of which
    is otherwise only weakly self-connected; all causal traffic between the
    periphery is funnelled through A. The figure shows the cause-effect
    structures of the subsystems {A, B} and {C} in the canonical state
    (1, 0, 0, 0, 0, 0).
    """
    s = 0.1
    w = 1 - s  # strong fan connection (0.9)
    j = 0.22  # B -> A return connection
    back = (1 - j) / 4  # C-F -> A return connections (0.195)
    # fmt: off
    weights = np.array([
        [0, w, w, w, w, w],
        [j, s, 0, 0, 0, 0],
        [back, 0, s, 0, 0, 0],
        [back, 0, 0, s, 0, 0],
        [back, 0, 0, 0, s, 0],
        [back, 0, 0, 0, 0, s],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 6, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig6a_system():
    return System(
        iit4_2023_fig6a_substrate(),
        state=(1, 0, 0, 0, 0, 0),
        node_indices=tuple(range(6)),
    )


@register_example
def iit4_2023_fig6b_substrate():
    """The 6-unit "modular" architecture of Fig 6B.

    Three strongly self-coupled pairs ({A, B}, {C, D}, {E, F}) with sparse weak
    links between modules. The figure shows the cause-effect structures of the
    subsystems {A, B}, {C, D} and {E, F} in the canonical state (1, 0, 0, 0, 0, 0).
    """
    s = 0.6  # within-module coupling
    m = 0.2  # self connection
    w = 0.1  # between-module coupling
    # fmt: off
    weights = np.array([
        [m, s, 0, w, 0, w],
        [s, m, 0, 0, 0, 0],
        [0, w, m, s, 0, w],
        [0, 0, s, m, 0, 0],
        [0, w, 0, w, m, s],
        [0, 0, 0, 0, s, m],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 6, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig6b_system():
    return System(
        iit4_2023_fig6b_substrate(),
        state=(1, 0, 0, 0, 0, 0),
        node_indices=tuple(range(6)),
    )


@register_example
def iit4_2023_fig6c_substrate():
    """The 6-unit "copy" architecture of Fig 6C.

    A directed cycle in which six units are unidirectionally connected with
    weight 1.0: each unit copies the state of the unit before it (A -> B -> ...
    -> F -> A), with some indeterminism. This is the only Fig 6 panel whose
    weights are given exactly in the paper's text (p. 32). The figure shows the
    cause-effect structure of the full system in the canonical state
    (1, 0, 0, 0, 0, 0).
    """
    # fmt: off
    weights = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 6, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig6c_system():
    return System(
        iit4_2023_fig6c_substrate(),
        state=(1, 0, 0, 0, 0, 0),
        node_indices=tuple(range(6)),
    )


@register_example
def iit4_2023_fig6d_substrate():
    """The 6-unit "specialized" architecture of Fig 6D.

    A densely connected network in which each unit has one strong input, two
    intermediate inputs and weak inputs from the rest, producing a highly
    integrated structure with a large number of high-degree relations. The figure
    shows the cause-effect structure of the full system in the canonical state
    (1, 0, 0, 0, 0, 0).
    """
    s = 0.7  # strong connection
    d = (1 - s) / 10  # weak connection (0.03)
    m = 3 * d  # self connection (0.09)
    w = 4 * d  # intermediate connection (0.12)
    # fmt: off
    weights = np.array([
        [m, s, d, d, d, w],
        [d, d, w, m, d, s],
        [d, m, d, s, w, d],
        [w, d, m, d, s, d],
        [s, w, d, d, m, d],
        [d, d, s, w, d, m],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 6, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig6d_system():
    return System(
        iit4_2023_fig6d_substrate(),
        state=(1, 0, 0, 0, 0, 0),
        node_indices=tuple(range(6)),
    )


@register_example
def iit4_2023_fig6e_substrate():
    """The 6-unit "structured" architecture of Fig 6E.

    A variant of the Fig 6D specialized network with the inputs to units C, D and
    F perturbed away from the regular pattern, breaking some of its symmetry. The
    figure shows the cause-effect structures of the subsystems {C, D} and
    {A, B, E, F} in the canonical state (1, 0, 0, 0, 0, 0).
    """
    s = 0.8  # strong connection
    d = (1 - s) / 10  # weak connection (0.02)
    m = 3 * d  # self connection (0.06)
    w = 4 * d  # intermediate connection (0.08)
    # fmt: off
    weights = np.array([
        [m, s, d,        d,        d,        w],
        [d, d, w,        m,        d,        s],
        [d, m, d + 0.2,  s - 0.2,  w,        d],
        [w, d, m + 0.2,  d + 0.2,  s - 0.4,  d],
        [s, w, d,        d,        m,        d],
        [d, d, s - 0.4,  w,        d + 0.4,  m],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 6, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig6e_system():
    return System(
        iit4_2023_fig6e_substrate(),
        state=(1, 0, 0, 0, 0, 0),
        node_indices=tuple(range(6)),
    )


@register_example
def iit4_2023_fig7_substrate():
    """The 5-unit state-dependent network of Fig 7.

    A near-symmetric ring of five units with predominantly strong forward
    coupling, a few inhibitory (negative-weight) connections, and one perturbed
    input (D <- A, weight 0.4). Fig 7 illustrates state-dependence: the same
    substrate yields different cause-effect structures in different states. The
    figure analyses the full system {A, B, C, D, E} with unit E active, in state
    (1, 1, 0, 0, 1) (panel A), and inactive, in state (1, 1, 0, 0, 0) (panel B).
    Panel C, with E *inactivated*, is :func:`iit4_2023_fig7_inactivated_substrate`.
    """
    s = 0.8  # strong forward connection
    w = (1 - s) / 4  # weak connection (0.05)
    q = -w  # inhibitory connection (-0.05)
    m = 0.4  # perturbed D <- A connection
    # fmt: off
    weights = np.array([
        [w, s, w, w, q],
        [q, w, s, w, w],
        [w, q, w, s, w],
        [m, w, q, w, s],
        [s, w, w, q, w],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 5, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig7_system():
    return System(
        iit4_2023_fig7_substrate(), state=(1, 1, 0, 0, 1), node_indices=tuple(range(5))
    )


@register_example
def iit4_2023_fig7_inactivated_substrate():
    """The Fig 7 network with unit E *inactivated* (Fig 7C).

    Inactivating a unit abolishes its cause-effect power: its state is frozen and
    folded into the dynamics of the remaining units (it has no counterfactual
    states and cannot be intervened upon). This is distinct from holding E as a
    *background condition* of a candidate system -- it is a lesion of the
    substrate itself. Here E is frozen in its OFF state, conditioning the Fig 7
    substrate's transition probabilities so that E's (strong, weight-0.8) input to
    A becomes a fixed bias; the first-maximal complex then shrinks to {A, B, C, D}.
    """
    return Substrate.from_factored(
        iit4_2023_fig7_substrate().factored_tpm.condition({4: 0}),
        node_labels=("A", "B", "C", "D", "E"),
    )


@register_example
def iit4_2023_fig7_inactivated_system():
    return System(
        iit4_2023_fig7_inactivated_substrate(),
        state=(1, 1, 0, 0, 0),
        node_indices=(0, 1, 2, 3),
    )
