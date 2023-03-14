#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm.py

import numpy as np
import pickle
import pytest

from pyphi import Subsystem, ExplicitTPM
from pyphi.tpm import reconstitute_tpm

@pytest.mark.parametrize(
    "tpm",
    [ExplicitTPM(np.random.rand(42)), ExplicitTPM(np.arange(42))]
)
def test_serialization(tpm):
    assert tpm.array_equal(pickle.loads(pickle.dumps(tpm)))


def test_np_operations():
    # fmt: off
    tpm = ExplicitTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = tpm * tpm
    # fmt: off
    expected = ExplicitTPM(
        np.array([
            [9, 9],
            [9, 9]
        ])
    )
    # fmt: on

    assert actual.array_equal(expected)


def test_array_ufunc():
    # fmt: off
    tpm = ExplicitTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = np.multiply(tpm, tpm)
    # fmt: off
    expected = ExplicitTPM(
        np.array([
            [9, 9],
            [9, 9]
        ])
    )
    # fmt: on

    assert expected.array_equal(actual)


def test_getattr():
    tpm = ExplicitTPM(np.array([[0, 1]]))
    actual = tpm.real
    expected = np.array([[0, 1]])

    assert actual.all() == expected.all()

    # fmt: off
    tpm = ExplicitTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = tpm.sum(axis=0)
    expected = ExplicitTPM(np.array([6, 6]))

    assert expected.array_equal(expected)


def test_is_state_by_state():
    # State-by-state
    tpm = ExplicitTPM(np.ones((8, 8)))
    assert tpm.is_state_by_state()

    # State-by-node, multidimensional
    tpm = ExplicitTPM(np.ones((2, 2, 2, 3)))
    assert not tpm.is_state_by_state()

    # State-by-node, 2-dimensional
    tpm = ExplicitTPM(np.ones((8, 3)))
    assert not tpm.is_state_by_state()


def test_expand_tpm():
    tpm = np.ones((2, 1, 2))
    tpm[(0, 0)] = (0, 1)
    tpm = ExplicitTPM(tpm)
    # fmt: off
    answer = ExplicitTPM(
        np.array([
            [[0, 1],
             [0, 1]],
            [[1, 1],
             [1, 1]],
        ])
    )
    # fmt: on
    assert tpm.expand_tpm().array_equal(answer)


def test_marginalize_out(s):
    marginalized_distribution = s.tpm.marginalize_out([0])
    # fmt: off
    answer = ExplicitTPM(
        np.array([
            [[[0.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]],
             [[1.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])
    )

    # fmt: on
    assert marginalized_distribution.array_equal(answer)

    marginalized_distribution = s.tpm.marginalize_out([0, 1])
    # fmt: off
    answer = ExplicitTPM(
        np.array([
            [[[0.5, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])
    )
    # fmt: on
    assert marginalized_distribution.array_equal(answer)


def test_infer_cm(rule152):
    assert np.array_equal(rule152.tpm.infer_cm(), rule152.cm)


def test_reconstitute_tpm(standard, s_complete, rule152, noised):
    # Check subsystem and network TPM are the same when the subsystem is the
    # whole network
    assert np.array_equal(
        np.asarray(reconstitute_tpm(s_complete)),
        np.asarray(reconstitute_tpm(standard.tpm))
    )

    # Regression tests
    # fmt: off
    answer = np.array([
        [[[0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 1.],
          [0., 1., 0.]]],
        [[[0., 1., 0.],
          [0., 0., 0.]],
         [[1., 0., 1.],
          [1., 1., 0.]]],
    ])
    # fmt: on
    subsystem = Subsystem(rule152, (0,) * 5, (0, 1, 2))
    assert np.array_equal(answer, np.asarray(reconstitute_tpm(subsystem)))

    subsystem = Subsystem(noised, (0, 0, 0), (0, 1))
    # fmt: off
    answer = np.array([
        [[0. , 0. ],
         [0.7, 0. ]],
        [[0. , 0. ],
         [1. , 0. ]],
    ])
    # fmt: on
    assert np.array_equal(answer, np.asarray(reconstitute_tpm(subsystem)))
