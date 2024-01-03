#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm.py

import numpy as np
import pickle
import pytest
import random

from pyphi import examples, Network, Subsystem
from pyphi.convert import to_md
from pyphi.distribution import normalize
from pyphi.tpm import ExplicitTPM, reconstitute_tpm


@pytest.fixture()
def implicit_tpm(size, degree, node_states, seed=1337, deterministic=False):
    if degree > size:
        raise ValueError(
            f"The number of parents of each node (degree={degree}) cannot be"
            f"smaller than the size of the network ({size})."
        )
    if node_states < 2:
        raise ValueError("Nodes must have at least 2 node_states.")

    rng = random.Random(seed)

    def random_deterministic_repertoire():
        """Assign all probability to a single purview state at random."""
        repertoire = rng.sample([1] + (node_states - 1) * [0], node_states)
        return repertoire

    def random_repertoire(deterministic):
        if deterministic:
            return random_deterministic_repertoire()

        repertoire = np.array([rng.uniform(0, 1) for s in range(node_states)])
        # Normalize using L1 metric.
        return normalize(repertoire)

    tpm = []

    for node_index in range(size):
        # Generate |node_states| repertoires for each combination of parent
        # states at t - 1.
        node_tpm = [
            random_repertoire(deterministic)
            for j in range(node_states ** degree)
        ]

        # Select |degree| nodes at random as parents to this node, then reshape
        # node TPM to multidimensional form.
        node_shape = np.ones(size, dtype=int)
        parents = rng.sample(range(size), degree)
        node_shape[parents] = node_states

        node_tpm = np.array(node_tpm).reshape(tuple(node_shape) + (node_states,))

        tpm.append(node_tpm)

    return tpm


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
    answer = np.array([
            [[[0.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]],
             [[1.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])

    # fmt: on
    assert np.array_equal(
        np.asarray(reconstitute_tpm(marginalized_distribution)), answer
    )

    marginalized_distribution = s.tpm.marginalize_out([0, 1])
    # fmt: off
    answer = np.array([
            [[[0.5, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])
    # fmt: on
    assert np.array_equal(
        np.asarray(reconstitute_tpm(marginalized_distribution)), answer
    )


def test_infer_cm(rule152):
    assert np.array_equal(rule152.tpm.infer_cm(), rule152.cm)


def test_backward_tpm():
    network = examples.functionally_equivalent()
    implicit_tpm = network.tpm
    explicit_tpm = reconstitute_tpm(network.tpm)

    state = (1, 0, 0)

    # Backward TPM of full network must equal forward TPM.
    subsystem_indices = (0, 1, 2)

    backward = explicit_tpm.backward_tpm(state, subsystem_indices)
    assert backward.array_equal(explicit_tpm)

    backward = reconstitute_tpm(
        implicit_tpm.backward_tpm(state, subsystem_indices)
    )
    assert backward.array_equal(explicit_tpm)

    # Backward TPM of proper subsystem.
    # fmt: off
    answer = ExplicitTPM(
        np.array(
            [[[[1, 0, 0,]],
              [[1, 1, 1,]]],
             [[[0, 1, 0,]],
              [[0, 1, 1,]]]],
        )
    )
    # fmt: on
    subsystem_indices = (0, 1)

    backward = explicit_tpm.backward_tpm(state, subsystem_indices)
    assert backward.array_equal(answer)

    backward = reconstitute_tpm(
        implicit_tpm.backward_tpm(state, subsystem_indices)
    )
    assert backward.array_equal(answer)


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
