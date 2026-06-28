"""Tests for the composite-unit combination strategies."""

import numpy as np
import pytest

from pyphi.substrate_generator import mechanism_combinations as C


def test_selective_picks_farthest_from_half():
    assert C.selective([0.9, 0.2, 0.6]) == 0.9
    assert C.selective([0.4, 0.55]) == 0.4  # |0.4-0.5|=0.1 > |0.55-0.5|=0.05


def test_average_and_maximal():
    assert C.average([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert C.maximal([0.2, 0.9, 0.6]) == 0.9


def test_integrator_clamps_to_unit_interval():
    assert C.integrator([0.2, 0.3]) == pytest.approx(0.5)
    assert C.integrator([0.6, 0.7]) == 1.0
    assert C.integrator([0.0, 0.0]) == 0.0


def test_serial_is_one_minus_product_of_complements():
    assert C.serial([0.5, 0.5]) == pytest.approx(0.75)
    assert C.serial([0.2, 0.3, 0.4]) == pytest.approx(1 - 0.8 * 0.7 * 0.6)


def test_first_necessary_boosts_only_above_half():
    # Primary <= 0.5: returned unchanged.
    assert C.first_necessary([0.3, 0.9]) == 0.3
    # Primary > 0.5: boosted toward 1 (more when the others are inactive).
    boosted = C.first_necessary([0.8, 0.0])
    assert 0.8 < boosted <= 1.0


def test_first_necessary_constants_are_parametrized():
    """The original's fixed 5 / 0.5 are the defaults; both are tunable."""
    default = C.first_necessary([0.8, 0.1])
    explicit = C.first_necessary([0.8, 0.1], steepness=5.0, offset=0.5)
    assert default == explicit
    # A different steepness changes the boost.
    assert C.first_necessary([0.8, 0.1], steepness=50.0) != default


def test_composite_evaluates_and_combines_subspecs():
    weights = np.zeros((3, 3))
    weights[1, 0] = weights[2, 0] = 1.0
    comp = C.composite(
        [{"mechanism": "and", "inputs": (1, 2)}, {"mechanism": "or", "inputs": (1, 2)}],
        "maximal",
    )
    # max(and, or): or dominates whenever any input is on.
    assert comp(0, weights, (0, 1, 0)) == 1.0
    assert comp(0, weights, (0, 0, 0)) == 0.0


def test_composite_accepts_callable_combination():
    weights = np.zeros((2, 2))
    weights[1, 0] = 1.0
    comp = C.composite(
        [{"mechanism": "copy", "inputs": (1,)}],
        combination=lambda probs, **_: probs[0],
    )
    assert comp(0, weights, (0, 1)) == 1.0
