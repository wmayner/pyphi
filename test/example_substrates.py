import numpy as np

import pyphi
from pyphi import utils
from pyphi.labels import NodeLabels
from pyphi.macro import Blackbox
from pyphi.macro import MacroSystem
from pyphi.substrate import Substrate
from pyphi.system import System

# TODO pass just the system (contains a reference to the substrate)

standard = pyphi.examples.basic_substrate
s = pyphi.examples.basic_system
s_state = pyphi.examples.basic_state()


def s_empty():
    net = standard()
    return System(net, s_state, ())


def s_single():
    net = standard()
    return System(net, s_state, (1,))


def subsys_n0n2():
    net = standard()
    return System(net, s_state, (0, 2))


def subsys_n1n2():
    net = standard()
    return System(net, s_state, (1, 2))


def s_complete():
    net = standard(cm=None)
    return System(net, s_state, range(net.size))


def noised():
    # fmt: off
    tpm = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.2, 0.8, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.3],
        [0.1, 1.0, 0.0],
    ])
    cm = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def s_noised():
    n = noised()
    state = (1, 0, 0)
    return System(n, state, range(n.size))


def noisy_selfloop_single():
    net = pyphi.examples.basic_noisy_selfloop_substrate()
    state = (1, 0, 0)
    return System(net, state, (1,))


s_about_to_be_on = (0, 1, 1)
s_just_turned_on = (1, 0, 0)
s_all_off = (0, 0, 0)


def simple(cm=False):
    """Simple 'AND' substrate.

    Diagram:

    |           +~~~~~~~+
    |    +~~~~~~+   A   |<~~~~+
    |    | +~~~>| (AND) +~~~+ |
    |    | |    +~~~~~~~+   | |
    |    | |                | |
    |    v |                v |
    |  +~+~+~~~~+      +~~~~~~+~+
    |  |   B    |<~~~~~+    C   |
    |  | (OFF)  +~~~~~>|  (OFF) |
    |  +~~~~~~~~+      +~~~~~~~~+

    TPM:

    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    |  Previous state ~~> Current state |
    |~~~~~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~~|
    |      A, B, C     |    A, B, C     |
    |~~~~~~~~~~~~~~~~~~+~~~~~~~~~~~~~~~~|
    |     {0, 0, 0}    |   {0, 0, 0}    |
    |     {0, 0, 1}    |   {0, 0, 0}    |
    |     {0, 1, 0}    |   {0, 0, 0}    |
    |     {0, 1, 1}    |   {1, 0, 0}    |
    |     {1, 0, 0}    |   {0, 0, 0}    |
    |     {1, 0, 1}    |   {0, 0, 0}    |
    |     {1, 1, 0}    |   {0, 0, 0}    |
    |     {1, 1, 1}    |   {0, 0, 0}    |
    +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    """
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    # fmt: on
    if cm is False:
        cm = None
    return Substrate(tpm, cm=cm)


def simple_subsys_all_off():
    net = simple()
    return System(net, s_all_off, range(net.size))


def simple_subsys_all_a_just_on():
    net = simple()
    a_just_turned_on = (1, 0, 0)
    return System(net, a_just_turned_on, range(net.size))


def big(cm=None):
    """Return a large substrate."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0],
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
        [1, 1, 1, 1, 1],
    ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def big_subsys_all():
    """Return the system associated with ``big``."""
    net = big()
    state = (1,) * 5
    return System(net, state, range(net.size))


big_subsys_all_complete = big_subsys_all


def big_subsys_0_thru_3():
    """Return a system consisting of the first 4 nodes of ``big``."""
    net = big()
    state = (1,) * 5
    return System(net, state, range(net.size)[:-1])


def reducible(cm=False):
    tpm = np.zeros([2] * 2 + [2])
    if cm is False:
        cm = np.array([[1, 0], [0, 1]])
    r = Substrate(tpm, cm=cm)
    state = (0, 0)
    # Return the full system
    return System(r, state, range(r.size))


def rule30(cm=False):
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    if cm is False:
        cm = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ])
    # fmt: on
    rule30 = Substrate(tpm, cm=cm)
    all_off = (0, 0, 0, 0, 0)
    return System(rule30, all_off, range(rule30.size))


def trivial():
    """Single-node substrate with a self-loop."""
    tpm = np.array([[1], [1]])
    cm = np.array([[1]])
    net = Substrate(tpm, cm=cm)
    state = (1,)
    return System(net, state, range(net.size))


def eight_node(cm=False):
    """Eight-node substrate."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    if cm is False:
        cm = np.array([
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
        ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def eights():
    net = eight_node()
    state = [0] * 8
    return System(net, state, range(net.size))


def eights_complete():
    net = eight_node(cm=None)
    state = [0] * 8
    return System(net, state, range(net.size))


def eight_node_sbs(cm=False):
    tpm = [[1] + ([0] * 255)] * 256
    # fmt: off
    if cm is False:
        cm = np.array([
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
        ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def rule152(cm=False):
    # fmt: off
    tpm = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ])
    if cm is False:
        cm = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def rule152_s():
    net = rule152()
    state = [0] * 5
    return System(net, state, range(net.size))


def rule152_s_complete():
    net = rule152(cm=None)
    state = [0] * 5
    return System(net, state, range(net.size))


def macro(cm=False):
    # fmt: off
    tpm = np.array([
        [0.8281, 0.0819, 0.0819, 0.0081],
        [0.0000, 0.0000, 0.9100, 0.0900],
        [0.0000, 0.9100, 0.0000, 0.0900],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ])
    if cm is False:
        cm = np.array([
            [1, 1],
            [1, 1],
        ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def macro_s():
    net = macro()
    state = [0] * 2
    return System(net, state, range(net.size))


def micro(cm=False):
    # fmt: off
    tpm = np.array([
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
        [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
        [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
        [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
    ])
    if cm is False:
        cm = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])
    # fmt: on
    return Substrate(tpm, cm=cm)


def micro_s():
    net = micro()
    state = [1] * 4
    return System(net, state, range(net.size))


def micro_s_all_off():
    net = micro()
    state = [0] * 4
    return System(net, state, range(net.size))


# TODO: move to pyphi.examples?
def propagation_delay():
    """The basic PyPhi system with COPY gates on each of the connections in
    the original substrate, blackboxed over two time steps.
    """
    nodes = 8
    tpm = np.zeros((2**nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[5] == 1 or ps[7] == 1:
            cs[0] = 1
        if ps[0] == 1:
            cs[1] = 1
        if ps[1] ^ ps[6]:
            cs[2] = 1
        if ps[2] == 1:
            cs[3] = 1
            cs[7] = 1
        if ps[3] == 1:
            cs[4] = 1
        if ps[4] == 1:
            cs[5] = 1
            cs[6] = 1
        tpm[psi, :] = cs

    # fmt: off
    cm = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ])
    # fmt: on

    # Current state has the OR gate ON
    cs = (1, 0, 0, 0, 0, 0, 0, 0)

    substrate = Substrate(tpm, cm)

    # Elements 1, 3, 5, 6, 7 are the COPY gates
    # 0, 2, and 4 correspond to the original OR, XOR, and COPY
    partition = ((0, 5, 7), (3, 4), (1, 2, 6))
    outputs = (0, 2, 4)
    blackbox = Blackbox(partition, outputs)

    # Over two time steps, the system is functionally the same as the basic
    # system
    time_scale = 2

    return MacroSystem(
        substrate, cs, substrate.node_indices, time_scale=time_scale, blackbox=blackbox
    )


# Permutation-equivalent pair for symmetry tests
# ================================================


def and_xor_substrate():
    """AND-XOR 2-node substrate. Node 0: AND(0,1), Node 1: XOR(0,1).

    Both nodes receive input from both nodes (all-ones CM).
    Deterministic transitions:
        (0,0) -> (0,0), (1,0) -> (0,1), (0,1) -> (0,1), (1,1) -> (1,0)
    """
    # fmt: off
    tpm = np.array([
        [0, 0],  # (0,0) -> (0,0)
        [0, 1],  # (1,0) -> (0,1)
        [0, 1],  # (0,1) -> (0,1)
        [1, 0],  # (1,1) -> (1,0)
    ])
    # fmt: on
    cm = np.ones((2, 2))
    return Substrate(tpm, cm=cm, node_labels=NodeLabels(("AND", "XOR"), tuple(range(2))))


def xor_and_substrate():
    """XOR-AND 2-node substrate (AND-XOR with nodes 0 and 1 permuted).

    Both nodes receive input from both nodes (all-ones CM).
    Deterministic transitions:
        (0,0) -> (0,0), (1,0) -> (1,0), (0,1) -> (1,0), (1,1) -> (0,1)
    """
    # fmt: off
    tpm = np.array([
        [0, 0],  # (0,0) -> (0,0)
        [1, 0],  # (1,0) -> (1,0)
        [1, 0],  # (0,1) -> (1,0)
        [0, 1],  # (1,1) -> (0,1)
    ])
    # fmt: on
    cm = np.ones((2, 2))
    return Substrate(tpm, cm=cm, node_labels=NodeLabels(("XOR", "AND"), tuple(range(2))))


# Substrate-exclusion cascade test substrates
# =============================================
#
# Substrates engineered to exercise distinct branches of the IIT 4.0
# substrate-exclusion cascade per Albantakis et al. 2023 S1 Text:
# disjoint complexes accepted at the same φ_s tier, and an overlapping
# symmetric clique whose Composition tie violates the exclusion
# postulate.


def dual_and_xor_substrate():
    """Two independent AND-XOR pairs as a single 4-node substrate.

    Nodes (0, 1) form one AND-XOR pair and nodes (2, 3) form an
    independent AND-XOR pair; the two pairs do not connect. Each pair
    is irreducible in isolation, and the 4-node system is reducible
    (not strongly connected). Used to exercise the substrate-exclusion
    cascade's disjoint-complex branch: both 2-node subsystems should
    be accepted as separate complexes at the same φ_s tier.
    """
    n = 4
    tpm = np.zeros((2**n, n))
    for i in range(2**n):
        s0 = (i >> 0) & 1
        s1 = (i >> 1) & 1
        s2 = (i >> 2) & 1
        s3 = (i >> 3) & 1
        tpm[i, 0] = s0 & s1
        tpm[i, 1] = s0 ^ s1
        tpm[i, 2] = s2 & s3
        tpm[i, 3] = s2 ^ s3
    cm = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    return Substrate(
        tpm,
        cm=cm,
        node_labels=NodeLabels(("A1", "X1", "A2", "X2"), tuple(range(4))),
    )


def symmetric_triple_substrate():
    """Three-node substrate with a node-permutation symmetry across all three pairs.

    Each node receives input from the other two; the update function is
    the same parity-style rule for every node, so the substrate's TPM
    is invariant under any permutation of node labels. As a result, the
    three 2-node subsystems ``(0,1)``, ``(0,2)``, ``(1,2)`` are
    equivalent under permutation, producing identical ``φ_s`` and
    identical ``Φ`` values. Used to exercise the substrate-exclusion
    cascade's "tied at Composition" branch: the overlap clique among
    the three pairs should be skipped (exclusion postulate violation).
    """
    n = 3
    tpm = np.zeros((2**n, n))
    for i in range(2**n):
        s = [(i >> k) & 1 for k in range(n)]
        for k in range(n):
            others = [s[j] for j in range(n) if j != k]
            tpm[i, k] = others[0] ^ others[1]
    cm = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    return Substrate(tpm, cm=cm, node_labels=NodeLabels(("A", "B", "C"), (0, 1, 2)))
