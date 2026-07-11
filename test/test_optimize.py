"""Tests for pyphi.optimize: black-box optimization over substrate weights."""

import numpy as np
import pytest

import pyphi
from pyphi.optimize import weight_axes
from pyphi.substrate_generator import ising

# The IIT 4.0 (2023) Fig. 1A substrate; STATE is reachable with positive φ_s.
FIG1A_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
STATE = (1, 0, 0)


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


def test_weight_axes_sets_indexed_entries_without_mutating():
    original = FIG1A_WEIGHTS.copy()
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    substrate = axis(np.array([0.55, 0.35]))
    # Base matrix untouched.
    np.testing.assert_array_equal(FIG1A_WEIGHTS, original)
    # The built substrate is a real Substrate carrying the varied weights.
    assert substrate.node_labels is not None
    # A different vector yields a different substrate (weights actually applied).
    other = axis(np.array([0.10, 0.90]))
    one = np.asarray(substrate.joint_tpm())
    two = np.asarray(other.joint_tpm())
    assert one.shape == two.shape
    assert not np.array_equal(one, two)
