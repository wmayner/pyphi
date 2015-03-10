#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi
from pyphi import utils, constants, models


sub = pyphi.examples.basic_subsystem()
mip = pyphi.compute.big_mip(sub)
cut = pyphi.models.Cut((0,), (1, 2))
s_cut = pyphi.subsystem.Subsystem(sub.node_indices, sub.network, cut,
                                  sub._mice_cache)


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


def test_bipartition():
    answer = [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)),
              ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    assert answer == utils.bipartition((1, 2, 3))
    # Test with empty input
    assert [] == utils.bipartition(())


def test_emd_same_distributions():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert utils.hamming_emd(a, b) == 0.0


def test_uniform_distribution():
    assert np.array_equal(utils.uniform_distribution(3),
                          (np.ones(8)/8).reshape([2]*3))


def test_cut_mechanism_indices():
    assert ((0, 1), (0, 2), (0, 1, 2)) == utils.cut_mechanism_indices(sub, cut)


def test_cut_concepts():
    answer = tuple(mip.unpartitioned_constellation[i] for i in [1, 2, 3])
    assert answer == utils.cut_concepts(s_cut.cut_matrix,
                                        mip.unpartitioned_constellation)


def test_uncut_concepts():
    answer = tuple(mip.unpartitioned_constellation[i] for i in [0])
    assert answer == utils.uncut_concepts(s_cut.cut_matrix,
                                          mip.unpartitioned_constellation)
