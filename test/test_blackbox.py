
import numpy as np
import pytest

from pyphi import compute, config, macro, models, Network, utils

# TODO: move these to examples.py


@pytest.mark.slow
def test_basic_propogation_delay(s):
    # Create basic network with propagation delay.
    # COPY gates on each of the connections in the original network.

    nodes = 8
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if (ps[5] == 1 or ps[7] == 1):
            cs[0] = 1
        if (ps[0] == 1):
            cs[1] = 1
        if (ps[1] ^ ps[6]):
            cs[2] = 1
        if (ps[2] == 1):
            cs[3] = 1
            cs[7] = 1
        if (ps[3] == 1):
            cs[4] = 1
        if (ps[4] == 1):
            cs[5] = 1
            cs[6] = 1
        tpm[psi, :] = cs

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

    # Current state has the OR gate ON
    cs = (1, 0, 0, 0, 0, 0, 0, 0)

    net = Network(tpm, connectivity_matrix=cm)

    # Elements 1, 3, 5, 6, 7 are the COPY gates
    # 0, 2, and 4 correspond to the original OR, COPY, XOR
    hidden_indices = (1, 3, 5, 6, 7)

    # Over two time steps, the system is functionally the same as the basic system
    time_step = 2

    bb_sub = macro.MacroSubsystem(net, cs, net.node_indices,
                                  time_scale=time_step,
                                  hidden_indices=hidden_indices)

    bb_mip = compute.big_mip(bb_sub)
    assert bb_mip.phi == 2.125
    assert bb_mip.cut == models.Cut((0, 1, 2, 3, 4, 5, 6), (7,))


@pytest.mark.slow
def test_basic_nor_or():
    # A system composed of NOR and OR (copy) gates, which mimics the basic
    # pyphi network

    nodes = 12
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if (ps[5] == 0 and ps[11] == 0):
            cs[0] = 1
        if (ps[0] == 0):
            cs[1] = 1
        if (ps[1] == 1):
            cs[2] = 1
        if (ps[11] == 0):
            cs[3] = 1
        if (ps[3] == 0):
            cs[4] = 1
        if (ps[4] == 1):
            cs[5] = 1
        if (ps[2] == 0):
            cs[6] = 1
        if (ps[5] == 0):
            cs[7] = 1
        if (ps[6] == 0 and ps[7] == 0):
            cs[8] = 1
        if (ps[2] == 0 and ps[5] == 0):
            cs[9] = 1
        if (ps[9] == 1):
            cs[10] = 1
        if (ps[8] == 0 and ps[10] == 0):
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

    network = Network(tpm, connectivity_matrix=cm)

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
        mip = compute.big_mip(sub)

    assert mip.phi == 1.958332
    assert mip.cut == models.Cut((6,), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11))
    # After performing the 'ONE_CUT_APPROXIMATION'
    # The cut disrupts half of the connection from A (OR) to C (XOR).
    # It is able to do this because A 'enters' C from two different locations


@pytest.mark.slow
def test_xor_propogation_delay():
    # Three interconnected XOR gates, with COPY gates along each connection
    # acting as propagation delays.

    nodes = 9
    tpm = np.zeros((2 ** nodes, nodes))

    for psi, ps in enumerate(utils.all_states(nodes)):
        cs = [0 for i in range(nodes)]
        if (ps[2] ^ ps[7]):
            cs[0] = 1
        if (ps[0] == 1):
            cs[1] = 1
            cs[8] = 1
        if (ps[1] ^ ps[5]):
            cs[3] = 1
        if (ps[3] == 1):
            cs[2] = 1
            cs[4] = 1
        if (ps[4] ^ ps[8]):
            cs[6] = 1
        if (ps[6] == 1):
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

    network = Network(tpm, connectivity_matrix=cm)

    partition = ((0, 2, 7), (1, 3, 5), (4, 6, 8))
    output_indices = (0, 3, 6)
    blackbox = macro.Blackbox(partition, output_indices)
    assert blackbox.hidden_indices == (1, 2, 4, 5, 7, 8)

    time = 2
    subsys = macro.MacroSubsystem(network, state, network.node_indices,
                                  blackbox=blackbox, time_scale=time)

    big_mip = compute.big_mip(subsys)
    assert big_mip.phi == 1.874999
    assert big_mip.cut == models.Cut((0,), (1, 2, 3, 4, 5, 6, 7, 8))
