#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_distance.py

import numpy as np
import pytest

from pyphi import config, distance


def test_hamming_matrix():
    answer = np.array([[0, 1, 1, 2, 1, 2, 2, 3],
                       [1, 0, 2, 1, 2, 1, 3, 2],
                       [1, 2, 0, 1, 2, 3, 1, 2],
                       [2, 1, 1, 0, 3, 2, 2, 1],
                       [1, 2, 2, 3, 0, 1, 1, 2],
                       [2, 1, 3, 2, 1, 0, 2, 1],
                       [2, 3, 1, 2, 1, 2, 0, 1],
                       [3, 2, 2, 1, 2, 1, 1, 0]]).astype(float)
    assert np.array_equal(distance._hamming_matrix(3), answer)


def test_large_hamming_matrix():
    n = distance._NUM_PRECOMPUTED_HAMMING_MATRICES + 1
    distance._hamming_matrix(n)


def test_emd_same_distributions():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert distance.hamming_emd(a, b) == 0.0


def test_emd_validates_distribution_shapes():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((3, 3, 3)) / 27
    with pytest.raises(ValueError):
        distance.hamming_emd(a, b)


def test_l1_distance():
    a = np.array([0, 1, 2])
    b = np.array([2, 2, 4.5])
    assert distance.l1(a, b) == 5.5


def test_entropy_difference():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert distance.entropy_difference(a, b) == 0

    a = np.array([0, 1, 2])
    b = np.array([2, 2, 4.5])
    assert distance.entropy_difference(a, b) == 0.54979494760874348


def test_kld():
    a = np.array([0, 1.0])
    b = np.array([0.5, 0.5])

    assert distance.kld(a, b) == 1


def test_psq2():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert distance.psq2(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert distance.psq2(a, b) == 0.7334585933443496


def test_mp2q():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert distance.mp2q(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert distance.mp2q(a, b) == 4


def test_klm():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert distance.klm(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert distance.klm(a, b) == 2.0


def test_default_measures():
    assert set(distance.measures.all()) == set([
        'EMD',
        'L1',
        'KLD',
        'ENTROPY_DIFFERENCE',
        'PSQ2',
        'MP2Q',
        'KLM',
        'BLD'])


def test_default_asymmetric_measures():
    assert set(distance.measures.asymmetric()) == set([
        'KLD',
        'MP2Q',
        'KLM',
        'BLD'])


def test_system_repertoire_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(MEASURE='KLD'):
        with pytest.raises(ValueError):
            distance.system_repertoire_distance(a, b)


def test_suppress_np_warnings():
    @distance.np_suppress()
    def divide_by_zero():
        np.ones((2,)) / np.zeros((2,))

    @distance.np_suppress()
    def multiply_by_nan():
        np.array([1, 0]) * np.log(0)

    # Try and trigger an error:
    with np.errstate(divide='raise', invalid='raise'):
        divide_by_zero()
        multiply_by_nan()
