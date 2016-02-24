#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import pyphi
from pyphi import convert, macro, models, utils
from pyphi.convert import (state_by_node2state_by_state as sbn2sbs,
                           state_by_state2state_by_node as sbs2sbn)


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

#answer_cm = np.array([
#    [0, 1],
#    [1, 0]
#])
# TODO: using universal connectivity until we clarify how to handle
# connectivity for hidden and coarse-grained elements.
answer_cm = np.ones((2, 2))

state = (0, 0, 0, 0)

network = pyphi.Network(tpm, connectivity_matrix=cm)

output_grouping = ((0, 1), (2, 3))
state_grouping = (((0, 1), (2,)), ((0, 1), (2,)))
coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
subsystem = macro.MacroSubsystem(network, state, network.node_indices,
                                 coarse_grain=coarse_grain)

cut = pyphi.models.Cut((0,), (1, 2, 3))
cut_subsystem = macro.MacroSubsystem(network, state, network.node_indices,
                                     cut=cut, coarse_grain=coarse_grain)


def test_macro_subsystem():
    subsystem = macro.MacroSubsystem(network, state, network.node_indices,
                                     coarse_grain=coarse_grain)
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
    cut_subsystem = subsystem.apply_cut(cut)
    answer_tpm = np.array([
        [0.09, 0.20083333],
        [0.09, 0.4225],
        [1., 0.20083333],
        [1., 0.4225]
    ])
    assert np.array_equal(cut_subsystem.connectivity_matrix, answer_cm)
    assert np.all(cut_subsystem.tpm.reshape([4]+[2], order='F') - answer_tpm
                  < pyphi.constants.EPSILON)


# Tests for purely temporal blackboxing
# =====================================

tpm_noise = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25]
])

tpm_copy = sbn2sbs(np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]))

tpm_copy2 = sbn2sbs(np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
]))

tpm_copy3 = sbn2sbs(np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]))

tpm_huge = np.array([[1 if i == j+1 else 0
                      for i in range(1000)]
                     for j in range(1000)])
tpm_huge[999, 0] = 1


def test_sparse_blackbox():
    assert np.array_equal(utils.sparse_time(tpm_huge, 1001), tpm_huge)


def test_dense_blackbox():
    assert np.array_equal(utils.dense_time(tpm_noise, 2), tpm_noise)
    assert np.array_equal(utils.dense_time(tpm_noise, 3), tpm_noise)


def test_cycle_blackbox():
    assert np.array_equal(utils.sparse_time(tpm_copy, 2), tpm_copy2)
    assert np.array_equal(utils.sparse_time(tpm_copy, 3), tpm_copy3)
    assert np.array_equal(utils.dense_time(tpm_copy, 2), tpm_copy2)
    assert np.array_equal(utils.dense_time(tpm_copy, 3), tpm_copy3)


def test_run_tpm():
    tpm = sbs2sbn(np.array([
        [0, 1],
        [1, 0],
    ]))
    answer = sbs2sbn(np.array([
        [1, 0],
        [0, 1],
    ]))
    assert np.array_equal(utils.run_tpm(tpm, 2), answer)

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
    answer = convert.to_n_dimensional(np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ]))
    assert np.array_equal(utils.run_tpm(tpm, 2), answer)


def test_init_subsystem_in_time(s):
    time_subsys = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                                       time_scale=2)
    answer_tpm = convert.to_n_dimensional(np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ]))
    answer_cm = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
    ])
    assert np.array_equal(time_subsys.tpm, answer_tpm)
    assert np.array_equal(time_subsys.connectivity_matrix, answer_cm)


def test_macro_cut_is_for_micro_indices(s):
    with pytest.raises(ValueError):
        macro.MacroSubsystem(s.network, s.state, s.node_indices,
                             hidden_indices=(2,), cut=models.Cut((0,), (1,)))


def test_subsystem_equality(s):
    macro_subsys = macro.MacroSubsystem(s.network, s.state, s.node_indices)
    assert s != macro_subsys  # Although, should they be?

    macro_subsys_t = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                                          time_scale=2)
    assert macro_subsys != macro_subsys_t

    macro_subsys_h = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                                          hidden_indices=(0,))
    assert macro_subsys != macro_subsys_h

    coarse_grain = macro.CoarseGrain(((0, 1), (2,)), (((0, 1), (2,)), ((0,), (1,))))
    macro_subsys_c = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                                          coarse_grain=coarse_grain)
    assert macro_subsys != macro_subsys_c


# Test MacroSubsystem initialization
# ===============================================================

def test_blackbox(s):
    ms = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                              hidden_indices=(0, 2))
    # Conditioned on hidden indices and squeezed
    assert np.array_equal(ms.tpm, np.array([[0], [0]]))
    # Universal connectivity
    assert np.array_equal(ms.cm, np.array([[1]]))
    # Reindexed
    assert ms.node_indices == (0,)
    assert ms.state == (0,)
    assert not ms.is_micro


def test_blackbox_external(s):
    # Which is the same if one of these indices is external
    ms = macro.MacroSubsystem(s.network, s.state, (1, 2), hidden_indices=(2,))
    assert np.array_equal(ms.tpm, np.array([[0], [0]]))
    assert np.array_equal(ms.cm, np.array([[1]]))
    assert ms.node_indices == (0,)
    assert ms.state == (0,)


def test_coarse_grain(s):
    coarse_grain = macro.CoarseGrain(output_grouping=((0, 1), (2,)),
                                     state_grouping=((((0, 1), (2,)), ((0,), (1,)))))
    ms = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                              coarse_grain=coarse_grain)
    answer_tpm = np.array(
        [[[0, 0.66666667],
          [1, 0.66666667]],
         [[0, 0],
          [1, 0]]])
    assert np.allclose(ms.tpm, answer_tpm)
    assert np.array_equal(ms.cm, np.ones((2, 2)))
    assert ms.node_indices == (0, 1)
    assert ms.state == (0, 0)


def test_blackbox_and_coarse_grain(s):
    coarse_grain = macro.CoarseGrain(output_grouping=((0, 2),),
                                     state_grouping=((((0, 1), (2,)),)))
    ms = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                              hidden_indices=(1,), coarse_grain=coarse_grain)
    assert np.array_equal(ms.tpm, np.array([[0], [1]]))
    assert np.array_equal(ms.cm, [[1]])
    assert ms.node_indices == (0,)
    assert ms.size == 1
    assert ms.state == (0,)


def test_blackbox_and_coarse_grain_external(s):
    # Larger, with external nodes, blackboxed and coarse-grained
    tpm = np.zeros((2 ** 6, 6))
    network = pyphi.Network(tpm)
    state = (0, 0, 0, 0, 0, 0)

    hidden_indices = (4,)
    output_grouping = ((1,), (2,), (3, 5))
    state_grouping = (((0,), (1,)), ((1,), (0,)), ((0,), (1, 2)))
    coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
    ms = macro.MacroSubsystem(network, state, (1, 2, 3, 4, 5),
                              hidden_indices=hidden_indices,
                              coarse_grain=coarse_grain)
    answer_tpm = np.array(
        [[[[0, 1, 0],
           [0, 1, 0]],
          [[0, 1, 0],
           [0, 1, 0]]],
         [[[0, 1, 0],
           [0, 1, 0]],
          [[0, 1, 0],
           [0, 1, 0]]]])
    assert np.array_equal(ms.tpm, answer_tpm)
    assert np.array_equal(ms.cm, np.ones((3, 3)))
    assert ms.node_indices == (0, 1, 2)
    assert ms.size == 3
    assert ms.state == (0, 1, 0)
