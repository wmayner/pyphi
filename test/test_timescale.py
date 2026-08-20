"""Value-based tests for pyphi.timescale."""

import numpy as np
import pytest

from pyphi.exceptions import ConditionallyDependentError
from pyphi.timescale import dense_time
from pyphi.timescale import run_cm
from pyphi.timescale import run_tpm
from pyphi.timescale import sparse
from pyphi.timescale import sparse_time


def test_sparse_density_threshold():
    # sparse() returns whether (#nonzero / size) > threshold.
    assert sparse(np.array([[1, 0], [0, 1]]), threshold=0.1)  # density 0.5 > 0.1
    assert not sparse(np.array([[1, 0], [0, 0]]), threshold=0.4)  # density 0.25 !> 0.4


def test_dense_time_matrix_power():
    m = np.array([[0, 1], [1, 0]])
    assert np.array_equal(dense_time(m, 2), np.eye(2))  # swap^2 == identity


def test_sparse_time_matches_dense():
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(sparse_time(m, 2), dense_time(m, 2))


def test_run_cm_powers_and_clamps_to_one():
    cm = np.array([[1, 1], [1, 1]])
    # cm^2 = [[2,2],[2,2]]; values > 1 are clamped back to 1.
    assert np.array_equal(run_cm(cm, 2), np.array([[1, 1], [1, 1]]))


def test_run_tpm_one_step_is_identity_roundtrip():
    # A deterministic 2-node state-by-node TPM; running it for 1 step is the
    # convert -> matrix_power(1) -> convert round-trip, which returns the input.
    tpm = np.array(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    assert np.allclose(run_tpm(tpm, 1), tpm)


def test_run_tpm_correlated_nodes_raise_at_two_steps():
    # Both nodes read node 0 with different noise, so the exact two-step
    # dynamics are conditionally dependent: no state-by-node TPM represents
    # them, and silently returning the lossy round-trip would be wrong
    # (max abs error ~0.043 on this TPM).
    tpm = np.array(
        [
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.9, 0.8], [0.9, 0.8]],
        ]
    )
    assert np.allclose(run_tpm(tpm, 1), tpm)  # single step is unaffected
    with pytest.raises(ConditionallyDependentError):
        run_tpm(tpm, 2)


def test_run_tpm_independent_nodes_return_exact_iterated_dynamics():
    # Node i depends only on node i, so the iterated dynamics stay
    # conditionally independent and each node's two-step transition is the
    # square of its own 2x2 Markov chain.
    p0 = np.array([[0.7, 0.3], [0.4, 0.6]])  # P(node 0 next | node 0 now)
    p1 = np.array([[0.8, 0.2], [0.1, 0.9]])  # P(node 1 next | node 1 now)

    def sbn(m0, m1):
        tpm = np.empty((2, 2, 2))
        for s0 in (0, 1):
            for s1 in (0, 1):
                tpm[s0, s1] = [m0[s0, 1], m1[s1, 1]]
        return tpm

    got = run_tpm(sbn(p0, p1), 2)
    expected = sbn(p0 @ p0, p1 @ p1)
    assert np.allclose(got, expected)
