#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cyphi.utils as utils
from cyphi.network import Network
from .example_networks import WithExampleNetworks


class TestUtils(WithExampleNetworks):

    def test_marginalize_out(self):
        marginalized_distribution = utils.marginalize_out(
            self.m_network.nodes[0], self.m_network.tpm)
        assert np.array_equal(marginalized_distribution,
                              np.array([[[[0.,  0.,  0.5],
                                          [1.,  1.,  0.5]],

                                         [[1.,  0.,  0.5],
                                          [1.,  1.,  0.5]]]]))

    def test_purview_max_entropy_distribution(self):
        # Individual setUp
        size = 3
        state = np.array([0, 1, 0])
        past_state = np.array([1, 1, 0])
        tpm = np.zeros([2] * size + [size]).astype(float)
        network = Network(tpm, state, past_state)

        max_ent = utils.max_entropy_distribution(network.nodes[0:2],
                                                 network)
        assert max_ent.shape == (2, 2, 1)
        assert np.array_equal(
            max_ent,
            np.divide(np.ones(4), 4).reshape((2, 2, 1)))
        assert max_ent[0][1][0] == 0.25

    def test_combs_for_1D_input(self):
        n, k = 3, 2
        data = np.arange(n)
        assert np.array_equal(
            utils.combs(data, k),
            np.asarray([[0, 1],
                        [0, 2],
                        [1, 2]]))

    def test_comb_indices(self):
        n, k = 3, 2
        data = np.arange(6).reshape(2, 3)
        assert np.array_equal(
            data[:, utils.comb_indices(n, k)],
            np.asarray([[[0, 1],
                         [0, 2],
                         [1, 2]],

                        [[3, 4],
                         [3, 5],
                         [4, 5]]]))

    def test_powerset(self):
        a = np.arange(2)
        assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]

    def test_hamming_matrix(self):
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

    def test_bitstring_index(self):
        array = np.arange(8)
        bitstring = '11110100'
        assert np.array_equal(utils._bitstring_index(array, bitstring),
                              np.array([0, 1, 2, 3, 5]))

    def test_bitstring_index_wrong_shape(self):
        array = np.arange(8).reshape(2,4)
        bitstring = bin(6).zfill(8)
        with self.assertRaises(ValueError):
            assert utils._bitstring_index(array, bitstring)

    def test_bitstring_index_mismatched_length(self):
        array = np.arange(8)
        bitstring = bin(6)[2:]
        with self.assertRaises(ValueError):
            assert utils._bitstring_index(array, bitstring)

    def test_bitstring_index_forgot_strip_b(self):
        array = np.arange(8)
        bitstring = bin(6).zfill(8)
        with self.assertRaises(ValueError):
            assert utils._bitstring_index(array, bitstring)

    def test_flip(self):
        assert (utils._flip('011010001') ==
                            '100101110')

    def test_bipartitions(self):
        b = list(utils.bipartitions(np.arange(4)))
        answer =  [(np.array([0]), np.array([1, 2, 3])),
                   (np.array([1]), np.array([0, 2, 3])),
                   (np.array([0, 1]), np.array([2, 3])),
                   (np.array([2]), np.array([0, 1, 3])),
                   (np.array([0, 2]), np.array([1, 3])),
                   (np.array([1, 2]), np.array([0, 3])),
                   (np.array([0, 1, 2]), np.array([3]))]
        assert all((b[i][0] == answer[i][0]).all() and
                   (b[i][1] == answer[i][1]).all() for i in range(len(b)))

    def test_emd_same_distributions(self):
        a = np.ones((2,2,2))/8
        b = np.ones((2,2,2))/8
        assert utils.emd(a,b) == 0.0

    def test_emd_different_shapes(self):
        a = np.ones((2,1,2))/4
        b = np.ones((2,2,2))/8
        assert utils.emd(a,b) == 3.0

    def test_emd_mismatched_size(self):
        a = np.ones((2,2,2,2))/16
        b = np.ones((2,2,2))/8
        with self.assertRaises(ValueError):
            assert utils.emd(a,b)
