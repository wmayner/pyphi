
import numpy as np

from pyphi import compute, macro, models, Network


def test_standard_blackbox(s):
    # Create basic network with propagation delay.
    # COPY gates on each of the connections in the original network.

    nodes = 8
    states = 2 ** nodes

    tpm = np.zeros((states, nodes))

    for psi in range(states):
        ps = tuple(map(int, bin(psi)[2:].zfill(nodes)[::-1]))
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
