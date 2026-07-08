"""Tests for the free per-node-marginal array operations."""

import numpy as np

from pyphi.core.tpm._node_ops import condition
from pyphi.core.tpm._node_ops import marginalize_out


def test_marginalize_out_uniform_average_keepdims():
    # 2 binary inputs, trailing size-2 node axis.
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    out = marginalize_out(arr, [0])
    # Axis 0 collapsed to a size-1 mean.
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(out, arr.sum(axis=0, keepdims=True) / 2)


def test_marginalize_out_multiple_axes():
    arr = np.random.default_rng(0).random((2, 2, 2, 2))
    out = marginalize_out(arr, [0, 1])
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out, arr.sum((0, 1), keepdims=True) / 4)


def test_marginalize_out_empty_is_identity():
    arr = np.arange(4, dtype=float).reshape(2, 2)
    out = marginalize_out(arr, [])
    np.testing.assert_array_equal(out, arr)


def test_condition_fixes_axis_and_keeps_ndim():
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    out = condition(arr, {0: 1})
    assert out.shape == (1, 2, 2)
    np.testing.assert_array_equal(out, arr[1][np.newaxis, ...])


def test_condition_skips_singleton_axis():
    arr = np.arange(4, dtype=float).reshape(1, 2, 2)
    out = condition(arr, {0: 0})
    np.testing.assert_array_equal(out, arr)
