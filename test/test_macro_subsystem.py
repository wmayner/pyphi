#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import pyphi
from pyphi import convert, macro, models, utils
from pyphi.convert import (state_by_node2state_by_state as sbn2sbs,
                           state_by_state2state_by_node as sbs2sbn)

@pytest.fixture()
def macro_subsystem():

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

    state = (0, 0, 0, 0)

    network = pyphi.Network(tpm, connectivity_matrix=cm)

    partition = ((0, 1), (2, 3))
    grouping = (((0, 1), (2,)), ((0, 1), (2,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)

    return macro.MacroSubsystem(network, state, network.node_indices,
                                     coarse_grain=coarse_grain)


def test_cut_indices(macro_subsystem, s):
    assert macro_subsystem.cut_indices == (0, 1, 2, 3)
    micro = macro.MacroSubsystem(s.network, s.state, s.node_indices)
    assert micro.cut_indices == (0, 1, 2)


def test_pass_node_indices_as_a_range(s):
    # Test that node_indices can be a `range`
    macro.MacroSubsystem(s.network, s.state, range(s.size))


#answer_cm = np.array([
#    [0, 1],
#    [1, 0]
#])
# TODO: using universal connectivity until we clarify how to handle
# connectivity for hidden and coarse-grained elements.
answer_cm = np.ones((2, 2))


def test_macro_subsystem(macro_subsystem):
    answer_tpm = np.array([
        [0.09, 0.09],
        [0.09, 1.],
        [1., 0.09],
        [1., 1.]
    ])
    assert np.array_equal(macro_subsystem.connectivity_matrix, answer_cm)
    assert np.all(macro_subsystem.tpm.reshape([4]+[2], order='F') - answer_tpm
                  < pyphi.constants.EPSILON)


def test_macro_cut_subsystem(macro_subsystem):
    cut = pyphi.models.Cut((0,), (1, 2, 3))
    cut_subsystem = macro_subsystem.apply_cut(cut)
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



def test_sparse_blackbox():
    tpm_huge = np.array([[1 if i == j+1 else 0
                          for i in range(1000)]
                         for j in range(1000)])
    tpm_huge[999, 0] = 1
    assert np.array_equal(utils.sparse_time(tpm_huge, 1001), tpm_huge)


def test_dense_blackbox():
    tpm_noise = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])
    assert np.array_equal(utils.dense_time(tpm_noise, 2), tpm_noise)
    assert np.array_equal(utils.dense_time(tpm_noise, 3), tpm_noise)


def test_cycle_blackbox():
    tpm = sbn2sbs(np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1]
    ]))

    # tpm over 2 timesteps
    tpm2 = sbn2sbs(np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]))

    # tpm over 3 timesteps
    tpm3 = sbn2sbs(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]))

    assert np.array_equal(utils.sparse_time(tpm, 2), tpm2)
    assert np.array_equal(utils.sparse_time(tpm, 3), tpm3)
    assert np.array_equal(utils.dense_time(tpm, 2), tpm2)
    assert np.array_equal(utils.dense_time(tpm, 3), tpm3)


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
                             blackbox=macro.Blackbox((2,), (0, 1)),
                             cut=models.Cut((0,), (1,)))


def test_subsystem_equality(s):
    state = (0, 0, 0)
    macro_subsys = macro.MacroSubsystem(s.network, state, s.node_indices)
    assert s != macro_subsys  # Although, should they be?

    macro_subsys_t = macro.MacroSubsystem(s.network, state, s.node_indices,
                                          time_scale=2)
    assert macro_subsys != macro_subsys_t

    blackbox = macro.Blackbox(((0, 1, 2),), (2,))
    macro_subsys_h = macro.MacroSubsystem(s.network, state, s.node_indices,
                                          blackbox=blackbox)
    assert macro_subsys != macro_subsys_h

    coarse_grain = macro.CoarseGrain(((0, 1), (2,)), (((0, 1), (2,)), ((0,), (1,))))
    macro_subsys_c = macro.MacroSubsystem(s.network, state, s.node_indices,
                                          coarse_grain=coarse_grain)
    assert macro_subsys != macro_subsys_c


# Test MacroSubsystem initialization
# ===============================================================

def test_blackbox(s):
    ms = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                              blackbox=macro.Blackbox(((0, 1, 2),), (1,)))
    # Conditioned on hidden indices and squeezed
    assert np.array_equal(ms.tpm, np.array([[0], [0]]))
    # Universal connectivity
    assert np.array_equal(ms.cm, np.array([[1]]))
    # Reindexed
    assert ms.node_indices == (0,)
    assert ms.state == (0,)


def test_blackbox_external(s):
    # Which is the same if one of these indices is external
    ms = macro.MacroSubsystem(s.network, s.state, (1, 2),
                              blackbox=macro.Blackbox(((1, 2),), (1,)))
    assert np.array_equal(ms.tpm, np.array([[0], [0]]))
    assert np.array_equal(ms.cm, np.array([[1]]))
    assert ms.node_indices == (0,)
    assert ms.state == (0,)


def test_coarse_grain(s):
    coarse_grain = macro.CoarseGrain(partition=((0, 1), (2,)),
                                     grouping=((((0, 1), (2,)), ((0,), (1,)))))
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
    blackbox = macro.Blackbox(((0, 1, 2),), (0, 2))
    coarse_grain = macro.CoarseGrain(partition=((0, 2),),
                                     grouping=((((0, 1), (2,)),)))
    ms = macro.MacroSubsystem(s.network, s.state, s.node_indices,
                              blackbox=blackbox, coarse_grain=coarse_grain)
    assert np.array_equal(ms.tpm, np.array([[0], [1]]))
    assert np.array_equal(ms.cm, [[1]])
    assert ms.node_indices == (0,)
    assert ms.size == 1
    assert ms.state == (0,)


def test_blackbox_and_coarse_grain_external():
    # Larger, with external nodes, blackboxed and coarse-grained
    tpm = np.zeros((2 ** 6, 6))
    network = pyphi.Network(tpm)
    state = (0, 0, 0, 0, 0, 0)

    blackbox = macro.Blackbox(((1, 4), (2,), (3,), (5,),), (1, 2, 3, 5))
    partition = ((1,), (2,), (3, 5))
    grouping = (((0,), (1,)), ((1,), (0,)), ((0,), (1, 2)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    ms = macro.MacroSubsystem(network, state, (1, 2, 3, 4, 5),
                              blackbox=blackbox, coarse_grain=coarse_grain)
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


@pytest.mark.slow
def test_blackbox_emergence():
    network = pyphi.examples.macro_network()
    state = (0, 0, 0, 0)
    result = macro.emergence(network, state, blackbox=True,
                             coarse_grain=True, time_scales=[1, 2])
    assert result.phi == 0.713678
    assert result.emergence == 0.599789
