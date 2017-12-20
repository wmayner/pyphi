import numpy as np
import pytest

from pyphi import Network, compute, config, convert, macro, models, utils


# TODO: move these to examples.py
@pytest.fixture
def degenerate():

    nodes = 6
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[5] == 1:
            cs[0] = 1
            cs[1] = 1
        if ps[0] == 1 and ps[1]:
            cs[2] = 1
        if ps[2] == 1:
            cs[3] = 1
            cs[4] = 1
        if ps[3] == 1 and ps[4] == 1:
            cs[5] = 1
        tpm[psi, :] = cs

    cm = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0]
    ])

    current_state = (0, 0, 0, 0, 0, 0)

    network = Network(tpm, cm)

    partition = ((0, 1, 2), (3, 4, 5))
    output_indices = (2, 5)
    blackbox = macro.Blackbox(partition, output_indices)
    time_scale = 2

    return macro.MacroSubsystem(network, current_state, network.node_indices,
                                blackbox=blackbox, time_scale=time_scale)


@pytest.mark.veryslow
def test_basic_nor_or():
    # A system composed of NOR and OR (copy) gates, which mimics the basic
    # pyphi network

    nodes = 12
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[5] == 0 and ps[11] == 0:
            cs[0] = 1
        if ps[0] == 0:
            cs[1] = 1
        if ps[1] == 1:
            cs[2] = 1
        if ps[11] == 0:
            cs[3] = 1
        if ps[3] == 0:
            cs[4] = 1
        if ps[4] == 1:
            cs[5] = 1
        if ps[2] == 0:
            cs[6] = 1
        if ps[5] == 0:
            cs[7] = 1
        if ps[6] == 0 and ps[7] == 0:
            cs[8] = 1
        if ps[2] == 0 and ps[5] == 0:
            cs[9] = 1
        if ps[9] == 1:
            cs[10] = 1
        if ps[8] == 0 and ps[10] == 0:
            cs[11] = 1
        tpm[psi, :] = cs

    cm = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    state = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    network = Network(tpm, cm=cm)

    # (0, 1, 2) compose the OR element,
    # (3, 4, 5) the COPY,
    # (6, 7, 8, 9, 10, 11) the XOR
    partition = ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9, 10, 11))
    output = (2, 5, 11)
    blackbox = macro.Blackbox(partition, output)
    assert blackbox.hidden_indices == (0, 1, 3, 4, 6, 7, 8, 9, 10)
    time = 3

    sub = macro.MacroSubsystem(network, state, network.node_indices,
                               blackbox=blackbox, time_scale=time)

    with config.override(CUT_ONE_APPROXIMATION=True):
        sia = compute.sia(sub)

    assert sia.phi == 1.958332
    assert sia.cut == models.Cut((6,), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11))
    # After performing the 'ONE_CUT_APPROXIMATION'
    # The cut disrupts half of the connection from A (OR) to C (XOR).
    # It is able to do this because A 'enters' C from two different locations


@pytest.mark.veryslow
def test_xor_propogation_delay():
    # Three interconnected XOR gates, with COPY gates along each connection
    # acting as propagation delays.

    nodes = 9
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[2] ^ ps[7]:
            cs[0] = 1
        if ps[0] == 1:
            cs[1] = 1
            cs[8] = 1
        if ps[1] ^ ps[5]:
            cs[3] = 1
        if ps[3] == 1:
            cs[2] = 1
            cs[4] = 1
        if ps[4] ^ ps[8]:
            cs[6] = 1
        if ps[6] == 1:
            cs[5] = 1
            cs[7] = 1
        tpm[psi, :] = cs

    cm = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0]
    ])

    # The state of the system is all OFF
    state = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    network = Network(tpm, cm=cm)

    partition = ((0, 2, 7), (1, 3, 5), (4, 6, 8))
    output_indices = (0, 3, 6)
    blackbox = macro.Blackbox(partition, output_indices)
    assert blackbox.hidden_indices == (1, 2, 4, 5, 7, 8)

    time = 2
    subsys = macro.MacroSubsystem(network, state, network.node_indices,
                                  blackbox=blackbox, time_scale=time)

    sia = compute.sia(subsys)
    assert sia.phi == 1.874999
    assert sia.cut == models.Cut((0,), (1, 2, 3, 4, 5, 6, 7, 8))


@pytest.mark.xfail
def test_soup():
    # An first example attempting to capture the "soup" metaphor
    #
    # The system will consist of 6 elements 2 COPY elements (A, B) input to an
    # AND element (C) AND element (C) inputs to two COPY elements (D, E) 2 COPY
    # elements (D, E) input to an AND element (F) AND element (F) inputs to two
    # COPY elements (A, B)
    #
    # For the soup example, element B receives an additional input from D, and
    # implements AND logic instead of COPY

    nodes = 6
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[5] == 1:
            cs[0] = 1
        if ps[3] == 1 and ps[5] == 1:
            cs[1] = 1
        if ps[0] == 1 and ps[1]:
            cs[2] = 1
        if ps[2] == 1:
            cs[3] = 1
            cs[4] = 1
        if ps[3] == 1 and ps[4] == 1:
            cs[5] = 1
        tpm[psi, :] = cs

    cm = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0]
    ])

    network = Network(tpm, cm)

    # State all OFF
    state = (0, 0, 0, 0, 0, 0)
    assert compute.major_complex(network, state).phi == 0.125

    # With D ON (E must also be ON otherwise the state is unreachable)
    state = (0, 0, 0, 1, 1, 0)
    assert compute.major_complex(network, state).phi == 0.215278

    # Once the connection from D to B is frozen (with D in the ON state), we
    # recover the degeneracy example
    state = (0, 0, 0, 1, 1, 0)
    partition = ((0, 1, 2), (3, 4, 5))
    output_indices = (2, 5)
    blackbox = macro.Blackbox(partition, output_indices)
    time = 2
    sub = macro.MacroSubsystem(network, state, (0, 1, 2, 3, 4, 5),
                               blackbox=blackbox, time_scale=time)
    assert compute.phi(sub) == 0.638888

    # When the connection from D to B is frozen (with D in the OFF state),
    # element B is inactivated and integration is compromised.
    state = (0, 0, 0, 0, 0, 0)
    partition = ((0, 1, 2), (3, 4, 5))
    output_indices = (2, 5)
    blackbox = macro.Blackbox(partition, output_indices)
    time = 2
    sub = macro.MacroSubsystem(network, state, (0, 1, 2, 3, 4, 5),
                               blackbox=blackbox, time_scale=time)
    assert compute.phi(sub) == 0


@pytest.mark.slow
def test_coarsegrain_spatial_degenerate():
    # TODO: move to docs?
    # macro-micro examples from Hoel2016
    # Example 2 - Spatial Degenerate
    # The micro system has a full complex, and big_phi = 0.19
    # The optimal coarse-graining groups AB, CD and EF, each with state
    # mapping ((0, 1), (2))

    nodes = 6
    tpm = np.zeros((2**nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if ps[0] == 1 and ps[1] == 1:
            cs[2] = 1
            cs[3] = 1
        if ps[2] == 1 and ps[3] == 1:
            cs[4] = 1
            cs[5] = 1
        if ps[4] == 1 and ps[5] == 1:
            cs[0] = 1
            cs[1] = 1
        tpm[psi, :] = cs

    cm = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0]
    ])

    state = (0, 0, 0, 0, 0, 0)

    net = Network(tpm, cm)

    mc = compute.major_complex(net, state)
    assert mc.phi == 0.194445

    partition = ((0, 1), (2, 3), (4, 5))
    grouping = (((0, 1), (2,)), ((0, 1), (2,)), ((0, 1), (2, )))
    coarse = macro.CoarseGrain(partition, grouping)

    sub = macro.MacroSubsystem(net, state, range(net.size),
                               coarse_grain=coarse)

    sia = compute.sia(sub)
    assert sia.phi == 0.834183


def test_degenerate(degenerate):
    assert np.array_equal(degenerate.tpm, convert.to_multidimensional(np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])))
    assert np.array_equal(degenerate.cm, np.array([
        [0, 1],
        [1, 0]
    ]))
    sia = compute.sia(degenerate)
    assert sia.phi == 0.638888


def test_basic_propagation_delay(s, propagation_delay):
    # bb_sia = compute.sia(bb_sub)
    # assert bb_sia.phi == 2.125
    # assert bb_sia.cut == models.Cut((0, 1, 2, 3, 4, 5, 6), (7,))

    assert np.array_equal(propagation_delay.cm, s.cm)
