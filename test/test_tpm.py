import pickle

import numpy as np
import pytest
from numpy.random import default_rng

from pyphi import JointTPM
from pyphi import System
from pyphi.tpm import reconstitute_tpm
from pyphi.tpm import simulate


@pytest.mark.parametrize("tpm", [JointTPM(np.random.rand(42)), JointTPM(np.arange(42))])
def test_serialization(tpm):
    assert tpm.array_equal(pickle.loads(pickle.dumps(tpm)))


def test_np_operations():
    # fmt: off
    tpm = JointTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = tpm * tpm
    # fmt: off
    expected = JointTPM(
        np.array([
            [9, 9],
            [9, 9]
        ])
    )
    # fmt: on

    assert actual.array_equal(expected)


def test_array_ufunc():
    # fmt: off
    tpm = JointTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = np.multiply(tpm, tpm)
    # fmt: off
    expected = JointTPM(
        np.array([
            [9, 9],
            [9, 9]
        ])
    )
    # fmt: on

    assert expected.array_equal(actual)


def test_getattr():
    tpm = JointTPM(np.array([[0, 1]]))
    actual = tpm.real
    expected = np.array([[0, 1]])

    assert actual.all() == expected.all()

    # fmt: off
    tpm = JointTPM(
        np.array([
            [3, 3],
            [3, 3]
        ])
    )
    # fmt: on
    actual = tpm.sum(axis=0)
    expected = JointTPM(np.array([6, 6]))

    assert expected.array_equal(actual)


def test_is_state_by_state():
    # State-by-state
    tpm = JointTPM(np.ones((8, 8)))
    assert tpm.is_state_by_state()

    # State-by-node, multidimensional
    tpm = JointTPM(np.ones((2, 2, 2, 3)))
    assert not tpm.is_state_by_state()

    # State-by-node, 2-dimensional
    tpm = JointTPM(np.ones((8, 3)))
    assert not tpm.is_state_by_state()


def test_expand_tpm():
    tpm = np.ones((2, 1, 2))
    tpm[(0, 0)] = (0, 1)
    tpm = JointTPM(tpm)
    # fmt: off
    answer = JointTPM(
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
    effect_arr = np.asarray(s.effect_tpm)
    for i in range(s.cause_tpm.n_nodes):
        np.testing.assert_array_equal(s.cause_tpm.factor(i)[..., 1], effect_arr[..., i])
    marginalized_distribution = s.effect_tpm.marginalize_out([0])
    # fmt: off
    answer = JointTPM(
        np.array([
            [[[0.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]],
             [[1.0, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])
    )

    # fmt: on
    assert marginalized_distribution.array_equal(answer)

    marginalized_distribution = s.effect_tpm.marginalize_out([0, 1])
    # fmt: off
    answer = JointTPM(
        np.array([
            [[[0.5, 0.0, 0.5],
              [1.0, 1.0, 0.5]]],
        ])
    )
    # fmt: on
    assert marginalized_distribution.array_equal(answer)


def test_infer_cm(rule152):
    from pyphi.tpm import JointTPM as _LegacyJointTPM

    legacy_tpm = _LegacyJointTPM(rule152.joint_tpm())
    assert np.array_equal(legacy_tpm.infer_cm(), rule152.cm)


def test_reconstitute_tpm(standard, s_complete, rule152, noised):
    # Check system and substrate TPM are the same when the system is the
    # whole substrate
    assert np.array_equal(reconstitute_tpm(s_complete), standard.joint_tpm())

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
    system = System(rule152, (0,) * 5, (0, 1, 2))
    assert np.array_equal(answer, reconstitute_tpm(system))

    system = System(noised, (0, 0, 0), (0, 1))
    # fmt: off
    answer = np.array([
        [[0. , 0. ],
         [0.7, 0. ]],
        [[0. , 0. ],
         [1. , 0. ]],
    ])
    # fmt: on
    assert np.array_equal(answer, reconstitute_tpm(system))


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
        simulate(standard.tpm, 0, 10, rng)
