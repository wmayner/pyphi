#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pyphi import macro, Network


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
    network = Network(tpm)
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
