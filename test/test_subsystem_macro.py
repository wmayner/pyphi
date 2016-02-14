#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi

tpm = np.zeros((16, 4)) + 0.3

tpm[12:, 0:2] = 1
tpm[3, 2:4] = 1
tpm[7, 2:4] = 1
tpm[11, 2:4] = 1
tpm[15, 2:4] = 1

cm = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0]
])

answer_cm = np.array([
    [0, 1],
    [1, 0]
])

state = (0, 0, 0, 0)

network = pyphi.Network(tpm, connectivity_matrix=cm)

output_grouping = ((0, 1), (2, 3))
state_grouping = (((0, 1), (2,)), ((0, 1), (2,)))

subsystem = pyphi.Subsystem(network, state, network.node_indices,
                            output_grouping=output_grouping,
                            state_grouping=state_grouping)

cut = pyphi.models.Cut((0,), (1, 2, 3))

cut_subsystem = pyphi.Subsystem(network, state, network.node_indices,
                                cut=cut,
                                output_grouping=output_grouping,
                                state_grouping=state_grouping)


def test_macro_subsystem():
    subsystem = pyphi.Subsystem(network, state, network.node_indices,
                                output_grouping=output_grouping,
                                state_grouping=state_grouping)
    answer_tpm = np.array([
        [0.09, 0.09],
        [0.09, 1.],
        [1., 0.09],
        [1., 1.]
    ])

    assert np.array_equal(subsystem.connectivity_matrix, answer_cm)
    assert np.all(subsystem.tpm.reshape([4]+[2], order='F') - answer_tpm
                  < pyphi.constants.EPSILON)


def test_macro_cut_subsystem():
    cut = pyphi.models.Cut((0,), (1, 2, 3))
    cut_subsystem = pyphi.Subsystem(network, state, network.node_indices,
                                    cut=cut,
                                    output_grouping=output_grouping,
                                    state_grouping=state_grouping)
    answer_tpm = np.array([
        [0.09, 0.20083333],
        [0.09, 0.4225],
        [1., 0.20083333],
        [1., 0.4225]
    ])
    assert np.array_equal(cut_subsystem.connectivity_matrix, answer_cm)
    assert np.all(cut_subsystem.tpm.reshape([4]+[2], order='F') - answer_tpm
                  < pyphi.constants.EPSILON)
