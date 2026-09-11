"""Tests for the ii(s) accessor and integrated fraction on the IIT 4.0 SIA."""

import pytest

import pyphi
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.utils import positive_part


@pytest.fixture(scope="module")
def fig1a_sia_2026():
    with config.override(**presets.iit4_2026, progress_bars=False):
        return examples.iit4_2023_fig1a_system().sia()


@pytest.fixture(scope="module")
def basic_sia_2026():
    with config.override(**presets.iit4_2026, progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def basic_sia_2023():
    with config.override(**presets.iit4_2023, progress_bars=False):
        return examples.basic_system().sia()


def test_ii_is_min_over_state_level_terms(fig1a_sia_2026):
    state = fig1a_sia_2026.system_state
    idiff = fig1a_sia_2026.intrinsic_differentiation
    expected = min(
        min(
            positive_part(float(state[direction].intrinsic_information)),
            positive_part(float(idiff[direction])),
        )
        for direction in idiff
    )
    assert fig1a_sia_2026.intrinsic_information == pytest.approx(expected)


def test_phi_bounded_by_ii_under_2026(fig1a_sia_2026):
    ii = fig1a_sia_2026.intrinsic_information
    assert ii is not None
    assert pyphi.numerics.is_positive(ii)
    assert float(fig1a_sia_2026.phi) <= ii + 10**-config.numerics.precision


def test_integrated_fraction_is_phi_over_ii(fig1a_sia_2026):
    fraction = fig1a_sia_2026.integrated_fraction
    ii = fig1a_sia_2026.intrinsic_information
    assert fraction == pytest.approx(float(fig1a_sia_2026.phi) / ii)
    assert 0.0 <= fraction <= 1.0


def test_zero_ii_forces_zero_phi_under_2026(basic_sia_2026):
    # The basic system is deterministic, so it specifies no intrinsic
    # differentiation and ii(s) = 0; the 2026 formalism then forces φ_s = 0.
    assert basic_sia_2026.intrinsic_information == pytest.approx(0.0)
    assert float(basic_sia_2026.phi) == pytest.approx(0.0)
    assert basic_sia_2026.integrated_fraction is None


def test_zero_ii_with_positive_phi_under_2023(basic_sia_2023):
    # Without the intrinsic-information requirement, φ_s can exceed ii(s);
    # with ii(s) = 0 there is no finite ratio, so the fraction is None.
    assert basic_sia_2023.intrinsic_information == pytest.approx(0.0)
    assert pyphi.numerics.is_positive(float(basic_sia_2023.phi))
    assert basic_sia_2023.integrated_fraction is None


def test_null_sia_accessors_are_none():
    null_sia = NullSystemIrreducibilityAnalysis()
    assert null_sia.intrinsic_information is None
    assert null_sia.integrated_fraction is None


def test_to_pandas_includes_ii_and_fraction(fig1a_sia_2026):
    record = fig1a_sia_2026.to_pandas()
    assert record["intrinsic_information"] == pytest.approx(
        fig1a_sia_2026.intrinsic_information
    )
    assert record["integrated_fraction"] == pytest.approx(
        fig1a_sia_2026.integrated_fraction
    )


def test_state_specification_intrinsic_specification_is_the_2023_intrinsic_information(
    fig1a_sia_2026,
):
    """Mayner et al. (2026) rename the 2023 per-state intrinsic information
    (Albantakis et al. 2023 Eqs. 5/7) to intrinsic specification (2026 Eqs. 7/9)."""
    for direction in Direction.both():
        spec = fig1a_sia_2026.system_state[direction]
        assert float(spec.intrinsic_specification) == float(spec.intrinsic_information)


def test_sia_intrinsic_specification_is_per_direction(fig1a_sia_2026):
    sia = fig1a_sia_2026
    spec = sia.intrinsic_specification
    assert set(spec) == set(Direction.both())
    for direction in Direction.both():
        assert spec[direction] == float(
            sia.system_state[direction].intrinsic_information
        )
    # Shape-parallel to the differentiation dict, and ii(s) is their joint minimum.
    assert set(sia.intrinsic_differentiation) == set(spec)
    terms = [positive_part(v) for v in spec.values()] + [
        positive_part(float(v)) for v in sia.intrinsic_differentiation.values()
    ]
    assert sia.intrinsic_information == pytest.approx(min(terms))


def test_sia_intrinsic_specification_none_on_null_sia():
    assert NullSystemIrreducibilityAnalysis().intrinsic_specification == {
        Direction.CAUSE: None,
        Direction.EFFECT: None,
    }


def test_sia_pandas_has_per_direction_specification_and_differentiation(fig1a_sia_2026):
    sia = fig1a_sia_2026
    record = sia.to_pandas()
    for column in (
        "cause_intrinsic_specification",
        "effect_intrinsic_specification",
        "cause_intrinsic_differentiation",
        "effect_intrinsic_differentiation",
    ):
        assert column in record.index
    assert record["cause_intrinsic_specification"] == pytest.approx(
        sia.intrinsic_specification[Direction.CAUSE]
    )
    assert record["effect_intrinsic_differentiation"] == pytest.approx(
        float(sia.intrinsic_differentiation[Direction.EFFECT])
    )
