#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyphi import utils, constants, models


def test_apply_cut():
    cm = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    cut = models.Cut(severed=(0, 3), intact=(1, 2))
    cut_cm = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0]
    ])
    assert np.array_equal(utils.apply_cut(cut, cm), cut_cm)


def test_fully_connected():
    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not utils.fully_connected(cm, (0,), (0, 1, 2))
    assert not utils.fully_connected(cm, (2,), (2,))
    assert not utils.fully_connected(cm, (0, 1), (1, 2))
    assert utils.fully_connected(cm, (0, 1), (0, 2))
    assert utils.fully_connected(cm, (1, 2), (1, 2))
    assert utils.fully_connected(cm, (0, 1, 2), (0, 1, 2))


def test_phi_eq():
    phi = 0.5
    close_enough = phi - constants.EPSILON/2
    not_quite = phi - constants.EPSILON*2
    assert utils.phi_eq(phi, close_enough)
    assert not utils.phi_eq(phi, not_quite)
    assert not utils.phi_eq(phi, (phi - phi))


def test_marginalize_out(s):
    marginalized_distribution = utils.marginalize_out(s.nodes[0].index,
                                                      s.network.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.,  0.,  0.5],
                                      [1.,  1.,  0.5]],
                                     [[1.,  0.,  0.5],
                                      [1.,  1.,  0.5]]]]))


def test_purview_max_entropy_distribution():
    max_ent = utils.max_entropy_distribution((0, 1), 3)
    assert max_ent.shape == (2, 2, 1)
    assert np.array_equal(max_ent,
                          (np.ones(4) / 4).reshape((2, 2, 1)))
    assert max_ent[0][1][0] == 0.25


def test_combs_for_1D_input():
    n, k = 3, 2
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k),
                          np.asarray([[0, 1],
                                      [0, 2],
                                      [1, 2]]))


def test_combs_r_is_0():
    n, k = 3, 0
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k), np.asarray([]))


def test_comb_indices():
    n, k = 3, 2
    data = np.arange(6).reshape(2, 3)
    assert np.array_equal(data[:, utils.comb_indices(n, k)],
                          np.asarray([[[0, 1],
                                       [0, 2],
                                       [1, 2]],
                                      [[3, 4],
                                       [3, 5],
                                       [4, 5]]]))


def test_powerset():
    a = np.arange(2)
    assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


def test_hamming_matrix():
    H = utils._hamming_matrix(3)
    answer = np.array([[0.,  1.,  1.,  2.,  1.,  2.,  2.,  3.],
                       [1.,  0.,  2.,  1.,  2.,  1.,  3.,  2.],
                       [1.,  2.,  0.,  1.,  2.,  3.,  1.,  2.],
                       [2.,  1.,  1.,  0.,  3.,  2.,  2.,  1.],
                       [1.,  2.,  2.,  3.,  0.,  1.,  1.,  2.],
                       [2.,  1.,  3.,  2.,  1.,  0.,  2.,  1.],
                       [2.,  3.,  1.,  2.,  1.,  2.,  0.,  1.],
                       [3.,  2.,  2.,  1.,  2.,  1.,  1.,  0.]])
    assert (H == answer).all()


def test_directed_bipartition():
    answer = [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)),
              ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    assert answer == utils.directed_bipartition((1, 2, 3))
    # Test with empty input
    assert [] == utils.directed_bipartition(())


def test_emd_same_distributions():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert utils.hamming_emd(a, b) == 0.0


def test_emd_validates_distribution_shapes():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((3, 3, 3)) / 27
    with pytest.raises(ValueError):
        utils.hamming_emd(a, b)


def test_l1_distance():
    a = np.array([0, 1, 2])
    b = np.array([2, 2, 4.5])
    assert utils.l1(a, b) == 5.5


def test_uniform_distribution():
    assert np.array_equal(utils.uniform_distribution(3),
                          (np.ones(8)/8).reshape([2]*3))


def test_block_cm():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0]
    ])
    cm5 = np.array([
        [1, 1],
        [0, 1]
    ])
    assert not utils.block_cm(cm1)
    assert utils.block_cm(cm2)
    assert utils.block_cm(cm3)
    assert not utils.block_cm(cm4)
    assert not utils.block_cm(cm5)


def test_block_reducible():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not utils.block_reducible(cm1, tuple(range(cm1.shape[0] - 1)),
                                     tuple(range(cm1.shape[1])))
    assert utils.block_reducible(cm2, (0, 1, 2), (0, 1, 2))
    assert utils.block_reducible(cm3, (0, 1), (0, 1, 2, 3, 4))
    assert not utils.block_reducible(cm4, (0, 1), (1, 2))


def test_get_inputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_inputs_from_cm(0, cm) == (1,)
    assert utils.get_inputs_from_cm(1, cm) == (0, 1)
    assert utils.get_inputs_from_cm(2, cm) == (1,)


def test_get_outputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_outputs_from_cm(0, cm) == (1,)
    assert utils.get_outputs_from_cm(1, cm) == (0, 1, 2)
    assert utils.get_outputs_from_cm(2, cm) == tuple()


def test_relevant_connections():
    cm = utils.relevant_connections(2, (0, 1), (1,))
    assert np.array_equal(cm, [
        [0, 1],
        [0, 1],
    ])
    cm = utils.relevant_connections(3, (0, 1), (0, 2))
    assert np.array_equal(cm, [
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 0],
    ])


def test_strongly_connected():
    # Strongly connected
    cm = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
    assert utils.strongly_connected(cm)

    # Disconnected
    cm = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]])
    assert not utils.strongly_connected(cm)

    # Weakly connected
    cm = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 1, 0]])
    assert not utils.strongly_connected(cm)

    # Nodes (0, 1) are strongly connected
    cm = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 0]])
    assert utils.strongly_connected(cm, (0, 1))


def test_normalize():
    x = np.array([2, 4, 2])
    assert np.array_equal(utils.normalize(x), np.array([.25, .5, .25]))
    x = np.array([[0, 4], [2, 2]])
    assert np.array_equal(utils.normalize(x), np.array([[0, .5], [.25, .25]]))
    x = np.array([0, 0])
    assert np.array_equal(utils.normalize(x), np.array([0, 0]))


def test_all_states():
    assert list(utils.all_states(0)) == []
    assert list(utils.all_states(1)) == [(0,), (1,)]
    assert list(utils.all_states(3)) == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]


def test_state_by_state():
    # State-by-state
    tpm = np.ones((8, 8))
    assert utils.state_by_state(tpm)

    # State-by-node, N-dimensional
    tpm = np.ones((2, 2, 2, 3))
    assert not utils.state_by_state(tpm)

    # State-by-node, 2-dimensional
    tpm = np.ones((8, 3))
    assert not utils.state_by_state(tpm)


def test_expand_tpm():
    tpm = np.ones((2, 1, 2))
    tpm[(0, 0)] = (0, 1)
    assert np.array_equal(utils.expand_tpm(tpm), np.array([
        [[0, 1],
         [0, 1]],
        [[1, 1],
         [1, 1]],
    ]))


def test_causally_significant_nodes():
    cm = np.array([
        [0, 0],
        [1, 0]
    ])
    assert utils.causally_significant_nodes(cm) == ()

    cm = np.array([
        [0, 1],
        [1, 0]
    ])
    assert utils.causally_significant_nodes(cm) == (0, 1)

    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    assert utils.causally_significant_nodes(cm) == (1, 2)


def test_marginal_zero():
    repertoire = np.array([
        [[0., 0.],
         [0., 0.5]],
        [[0., 0.],
         [0., 0.5]]])
    assert utils.marginal_zero(repertoire, 0) == 0.5
    assert utils.marginal_zero(repertoire, 1) == 0
    assert utils.marginal_zero(repertoire, 2) == 0


def test_marginal():
    repertoire = np.array([
        [[0., 0.],
         [0., 0.5]],
        [[0., 0.],
         [0., 0.5]]])
    assert np.array_equal(utils.marginal(repertoire, 0), np.array([[[0.5]], [[0.5]]]))
    assert np.array_equal(utils.marginal(repertoire, 1), np.array([[[0], [1]]]))
    assert np.array_equal(utils.marginal(repertoire, 2), np.array([[[0, 1]]]))


def test_independent():
    repertoire = np.array([
        [[ 0.25],
         [ 0.25]],
        [[ 0.25],
         [ 0.25]]])
    assert utils.independent(repertoire)

    repertoire = np.array([
        [[ 0.5],
         [ 0. ]],
        [[ 0. ],
         [ 0.5]]])
    assert not utils.independent(repertoire)


def test_purview_size(s):
    mechanisms = utils.powerset(s.node_indices)
    purviews = utils.powerset(s.node_indices)

    for mechanism, purview in zip(mechanisms, purviews):
        repertoire = s.cause_repertoire(mechanism, purview)
        assert utils.purview_size(repertoire) == len(purview)


def test_purview(s):
    mechanisms = utils.powerset(s.node_indices)
    purviews = utils.powerset(s.node_indices)

    for mechanism, purview in zip(mechanisms, purviews):
        repertoire = s.cause_repertoire(mechanism, purview)
        assert utils.purview(repertoire) == purview
