#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm.py

import numpy as np
import pytest
from numpy.random import default_rng

from pyphi import Subsystem
from pyphi.tpm import (
    expand_tpm,
    infer_cm,
    is_state_by_state,
    marginalize_out,
    reconstitute_tpm,
    simulate,
)


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
    # fmt: off
    answer = np.array([
        [[0, 1],
         [0, 1]],
        [[1, 1],
         [1, 1]],
    ])
    # fmt: on
    assert np.array_equal(expand_tpm(tpm), answer)


def test_marginalize_out(s):
    marginalized_distribution = marginalize_out([0], s.tpm)
    # fmt: off
    answer = np.array([
        [[[0.0, 0.0, 0.5],
          [1.0, 1.0, 0.5]],
         [[1.0, 0.0, 0.5],
          [1.0, 1.0, 0.5]]],
    ])
    # fmt: on
    assert np.array_equal(marginalized_distribution, answer)

    marginalized_distribution = marginalize_out([0, 1], s.tpm)
    # fmt: off
    answer = np.array([
        [[[0.5, 0.0, 0.5],
          [1.0, 1.0, 0.5]]],
    ])
    # fmt: on
    assert np.array_equal(marginalized_distribution, answer)


def test_infer_cm(rule152):
    assert np.array_equal(infer_cm(rule152.tpm), rule152.cm)


def test_reconstitute_tpm(standard, s_complete, rule152, noised):
    # Check subsystem and network TPM are the same when the subsystem is the
    # whole network
    assert np.array_equal(reconstitute_tpm(s_complete), standard.tpm)

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


def test_simulate_tpm_sanity():
    seed = 42
    rng = default_rng(seed)
    tpm = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]
    )
    path = simulate(tpm, 0, 10, rng)
    assert path == [0] + [3] * 9


def test_simulate_tpm():
    seed = 42
    rng = default_rng(seed)

    tpm = np.load("test/data/ising_tpm.npy")
    analytical_stationary_distribution = np.load(
        "test/data/ising_stationary_distribution.npy"
    )

    timesteps = 1e6
    initial_state = 0
    path = simulate(tpm, initial_state, timesteps, rng)
    counts, _ = np.histogram(path, bins=np.arange(tpm.shape[0] + 1))
    empirical_distribution = counts / timesteps

    assert np.allclose(
        empirical_distribution, analytical_stationary_distribution, atol=1e-3, rtol=0
    )


def test_simulate_tpm_requires_state_by_state(standard):
    seed = 42
    rng = default_rng(seed)

    with pytest.raises(ValueError):
        path = simulate(standard.tpm, 0, 10, rng)
