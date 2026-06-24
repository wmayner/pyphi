"""Value-based tests for pyphi.timescale."""

import numpy as np

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
