#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm.py

import numpy as np

from pyphi import Subsystem, ExplicitTPM
from pyphi.tpm import reconstitute_tpm


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
    assert np.array_equal(reconstitute_tpm(s_complete), standard.tpm.tpm)

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
    assert np.array_equal(answer, reconstitute_tpm(subsystem))

    subsystem = Subsystem(noised, (0, 0, 0), (0, 1))
    # fmt: off
    answer = np.array([
        [[0. , 0. ],
         [0.7, 0. ]],
        [[0. , 0. ],
         [1. , 0. ]],
    ])
    # fmt: on
    assert np.array_equal(answer, reconstitute_tpm(subsystem))
