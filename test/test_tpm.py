#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm.py

import numpy as np

from pyphi.tpm import expand_tpm, infer_cm, is_state_by_state, marginalize_out


def test_is_state_by_state():
    # State-by-state
    tpm = np.ones((8, 8))
    assert is_state_by_state(tpm)

    # State-by-node, multidimensional
    tpm = np.ones((2, 2, 2, 3))
    assert not is_state_by_state(tpm)

    # State-by-node, 2-dimensional
    tpm = np.ones((8, 3))
    assert not is_state_by_state(tpm)


def test_expand_tpm():
    tpm = np.ones((2, 1, 2))
    tpm[(0, 0)] = (0, 1)
    assert np.array_equal(expand_tpm(tpm), np.array([
        [[0, 1],
         [0, 1]],
        [[1, 1],
         [1, 1]],
    ]))


def test_marginalize_out(s):
    marginalized_distribution = marginalize_out([0], s.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.0, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]],
                                     [[1.0, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]]]]))

    marginalized_distribution = marginalize_out([0, 1], s.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.5, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]]]]))


def test_infer_cm(rule152):
    assert np.array_equal(infer_cm(rule152.tpm), rule152.cm)
