#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_metrics.py

import numpy as np
import pytest

from pyphi import config, metrics


def test_hamming_matrix():
    # fmt: off
    answer = np.array([
        [0, 1, 1, 2, 1, 2, 2, 3],
        [1, 0, 2, 1, 2, 1, 3, 2],
        [1, 2, 0, 1, 2, 3, 1, 2],
        [2, 1, 1, 0, 3, 2, 2, 1],
        [1, 2, 2, 3, 0, 1, 1, 2],
        [2, 1, 3, 2, 1, 0, 2, 1],
        [2, 3, 1, 2, 1, 2, 0, 1],
        [3, 2, 2, 1, 2, 1, 1, 0],
    ]).astype(float)
    # fmt: on
    assert np.array_equal(metrics._hamming_matrix(3), answer)


def test_large_hamming_matrix():
    n = metrics._NUM_PRECOMPUTED_HAMMING_MATRICES + 1
    metrics._hamming_matrix(n)


def test_emd_same_distributions():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert metrics.hamming_emd(a, b) == 0.0


def test_emd_validates_distribution_shapes():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((3, 3, 3)) / 27
    with pytest.raises(ValueError):
        metrics.hamming_emd(a, b)


def test_l1_metrics():
    a = np.array([0, 1, 2])
    b = np.array([2, 2, 4.5])
    assert metrics.l1(a, b) == 5.5


def test_entropy_difference():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert metrics.entropy_difference(a, b) == 0

    a = np.array([0, 1, 2])
    b = np.array([2, 2, 4.5])
    assert metrics.entropy_difference(a, b) == 0.54979494760874348


def test_kld():
    a = np.array([0, 1.0])
    b = np.array([0.5, 0.5])

    assert metrics.kld(a, b) == 1


def test_psq2():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert metrics.psq2(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert metrics.psq2(a, b) == 0.7334585933443496


def test_mp2q():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert metrics.mp2q(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert metrics.mp2q(a, b) == 4


def test_absolute_intrinsic_difference():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert metrics.absolute_intrinsic_difference(a, b) == 0

    a = np.array([[[1]], [[0]]])
    b = np.array([[[0.25]], [[0.75]]])
    assert metrics.absolute_intrinsic_difference(a, b) == 2.0


def test_default_measures():
    assert set(metrics.measures.all()) == set(
        [
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "AID",
            "KLM",
            "BLD",
            "ID",
            "PMI",
            "WPMI",
        ]
    )


def test_default_asymmetric_measures():
    assert set(metrics.measures.asymmetric()) == set(
        ["KLD", "MP2Q", "AID", "KLM", "BLD", "ID", "PMI", "WPMI"]
    )


def test_suppress_np_warnings():
    @metrics.np_suppress()
    def divide_by_zero():
        np.ones((2,)) / np.zeros((2,))

    @metrics.np_suppress()
    def multiply_by_nan():
        np.array([1, 0]) * np.log(0)

    # Try and trigger an error:
    with np.errstate(divide="raise", invalid="raise"):
        divide_by_zero()
        multiply_by_nan()
