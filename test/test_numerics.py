"""Tests for the tolerant scalar predicates in ``pyphi.numerics``."""

import numpy as np
import pytest

import pyphi
from pyphi import numerics

TOL = 10 ** (-13)  # default config.numerics.precision == 13


def test_eq_within_tolerance():
    assert numerics.eq(0.5, 0.5 + 1e-14)
    assert numerics.eq(0.5, 0.5 - 1e-14)
    assert not numerics.eq(0.5, 0.5 + 1e-12)


def test_eq_reads_config_at_call_time():
    with pyphi.config.override(precision=6):
        assert numerics.eq(0.5, 0.5 + 1e-9)
    assert not numerics.eq(0.5, 0.5 + 1e-9)


def test_is_zero():
    assert numerics.is_zero(0.0)
    assert numerics.is_zero(3.2e-16)  # summation-noise residue
    assert numerics.is_zero(-3.2e-16)
    assert not numerics.is_zero(1e-12)


def test_is_positive():
    assert numerics.is_positive(0.5)
    assert not numerics.is_positive(0.0)
    assert not numerics.is_positive(3.2e-16)  # noise is not positive
    assert not numerics.is_positive(-0.5)


def test_is_nonpositive():
    assert numerics.is_nonpositive(-0.5)
    assert numerics.is_nonpositive(0.0)
    assert not numerics.is_nonpositive(0.5)


def test_positive_mask_matches_elementwise_is_positive():
    rng = np.random.default_rng(20260710)
    values = np.concatenate(
        [
            rng.uniform(-1, 1, 100),
            np.array([0.0, 3.2e-16, -3.2e-16, 1e-13, 1e-12, -1e-12]),
        ]
    )
    mask = numerics.positive_mask(values)
    expected = np.array([numerics.is_positive(v) for v in values])
    np.testing.assert_array_equal(mask, expected)


def test_positive_mask_certain_node_surprisal():
    # -log2 of a float-noise near-1 probability is ~3e-16: mathematically
    # zero surprisal, must be masked out.
    surprisal = -np.log2(np.array([0.9999999999999998, 0.5, 0.25]))
    masked = surprisal[numerics.positive_mask(surprisal)]
    np.testing.assert_allclose(masked, [1.0, 2.0])


def test_round_to_precision():
    assert numerics.round_to_precision(0.5 + 4e-15) == round(0.5 + 4e-15, 13)
    with pyphi.config.override(precision=6):
        assert numerics.round_to_precision(0.1234567891) == pytest.approx(0.123457)


def test_eq_mask_matches_eq_elementwise():
    """eq_mask replicates eq (math.isclose semantics) exactly, including
    the non-finite special cases."""
    import numpy as np

    from pyphi import numerics

    values = np.array(
        [
            0.0,
            1e-14,
            1.0,
            1.0 - 1e-15,
            1e6,
            1e6 * (1 + 1e-14),
            -1.0,
            np.inf,
            -np.inf,
            np.nan,
        ]
    )
    for target in (1.0, 0.0, 1e6, -1.0, np.inf, np.nan):
        expected = [numerics.eq(v, target) for v in values]
        assert list(numerics.eq_mask(values, target)) == expected
