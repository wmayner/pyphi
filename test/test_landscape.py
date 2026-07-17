"""Tests for pyphi.landscape: parameter sections and local derivatives."""

import math

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi import exceptions
from pyphi import numerics
from pyphi.landscape import landscape_section
from pyphi.landscape import perturb
from pyphi.landscape import weight_axis
from pyphi.substrate_generator import ising
from test.conftest import skip_if_no_emd_backend

# The IIT 4.0 (2023) Fig. 1A substrate, with the A→B weight as the parameter
# axis. Known facts at the published point θ = 0.7 (default config): φ_s =
# 0.133873, partition margin 0.026941, cause state margin 0.003492, effect
# state margin 0.030059; a MIP switch (regime boundary) lies near θ ≈ 0.4527.
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


@pytest.fixture(scope="module")
def axis():
    return weight_axis([ising.probability] * 3, FIG1A_WEIGHTS, (0, 1), temperature=0.25)


def test_weight_axis_does_not_mutate_weights(axis):
    original = FIG1A_WEIGHTS.copy()
    axis(0.9)
    axis(0.1)
    np.testing.assert_array_equal(FIG1A_WEIGHTS, original)


def test_section_matches_direct_analyze(axis):
    grid = [0.55, 0.60, 0.65]
    section = landscape_section(axis, STATE, grid)
    assert list(section.df.index) == grid
    assert section.df.index.name == "theta"
    for theta, row in section.df.iterrows():
        sia = pyphi.analyze(axis(theta), STATE, compute="sia")
        assert row["phi"] == pytest.approx(float(sia.phi))
        assert row["signed_phi"] == pytest.approx(float(sia.signed_phi))
        assert row["normalized_phi"] == pytest.approx(float(sia.normalized_phi))
        assert row["partition_margin"] == pytest.approx(float(sia.partition_margin))
        assert row["cause_state"] == tuple(sia.system_state.cause.state)
        assert bool(row["effectively_tied"]) == sia.effectively_tied
    assert len(section.sias) == len(grid)
    assert section.skipped == []


def test_fig1a_published_point(axis):
    section = landscape_section(axis, STATE, [0.7])
    row = section.df.iloc[0]
    assert row["phi"] == pytest.approx(0.133873, abs=1e-6)
    assert row["partition_margin"] == pytest.approx(0.026941, abs=1e-6)
    assert row["cause_state_margin"] == pytest.approx(0.003492, abs=1e-6)
    assert row["effect_state_margin"] == pytest.approx(0.030059, abs=1e-6)
    assert not row["effectively_tied"]


def test_regime_boundary_detected(axis):
    section = landscape_section(axis, STATE, np.linspace(0.44, 0.46, 9))
    assert section.df["regime"].nunique() >= 2
    assert any(left < 0.4527 < right for left, right in section.boundaries)


def test_section_skips_unreachable():
    # A deterministic substrate analyzed at a state it can never reach:
    # every grid point is skipped.
    substrate = examples.basic_substrate()

    def build(_theta):
        return substrate

    section = landscape_section(build, (0, 1, 1), [0.0, 1.0])
    assert section.df.empty
    assert section.skipped == [0.0, 1.0]


def test_perturb_generic_point(axis):
    h = 1e-4
    result = perturb(axis, STATE, 0.60, h=h)
    lo = float(pyphi.analyze(axis(0.60 - h), STATE, compute="sia").signed_phi)
    hi = float(pyphi.analyze(axis(0.60 + h), STATE, compute="sia").signed_phi)
    assert result.derivative == pytest.approx((hi - lo) / (2 * h))
    assert result.left_derivative == pytest.approx(result.right_derivative, rel=1e-2)
    assert result.same_regime
    assert result.value == pytest.approx(
        float(pyphi.analyze(axis(0.60), STATE, compute="sia").signed_phi)
    )
    # Margin derivative matches a manual central difference.
    m_lo = float(pyphi.analyze(axis(0.60 - h), STATE, compute="sia").partition_margin)
    m_hi = float(pyphi.analyze(axis(0.60 + h), STATE, compute="sia").partition_margin)
    assert result.margin_derivatives["partition"] == pytest.approx(
        (m_hi - m_lo) / (2 * h)
    )


def test_perturb_signed_vs_clamped_in_plateau(axis):
    # At θ = 0.72 the raw integration is negative: the clamped phi is
    # exactly flat while the signed value still moves.
    clamped = perturb(axis, STATE, 0.72, quantity="phi")
    signed = perturb(axis, STATE, 0.72, quantity="signed_phi")
    assert clamped.derivative == pytest.approx(0.0)
    assert not numerics.eq(signed.derivative, 0.0)


def test_perturb_flags_regime_straddle(axis):
    result = perturb(axis, STATE, 0.4527, h=5e-3)
    assert not result.same_regime
    assert not math.isclose(result.left_derivative, result.right_derivative, rel_tol=0.5)


def test_switch_distance_matches_known_cliff(axis):
    # The specified-cause-state switch lies at θ ≈ 0.7016623 (bisected to
    # 1e-8), i.e. 0.0016623 above the published point in the A→B weight.
    # The linearized estimate agrees with the true distance to ~1%.
    result = perturb(axis, STATE, 0.7)
    distance = result.switch_distances["cause_state"]
    assert distance == pytest.approx(0.0016623, rel=0.01)


def test_perturb_rejects_unknown_quantity(axis):
    with pytest.raises(ValueError, match="unknown quantity"):
        perturb(axis, STATE, 0.6, quantity="big_phi")


def test_perturb_unreachable_raises():
    substrate = examples.basic_substrate()

    def build(_theta):
        return substrate

    with pytest.raises(exceptions.StateUnreachableError):
        perturb(build, (0, 1, 1), 0.5)


@skip_if_no_emd_backend
def test_landscape_section_iit3_rows_carry_defaults():
    """A documented non-4.0 preset produces rows, not AttributeErrors.

    IIT 4.0-only columns carry NaN/None, the contract established by
    ``pyphi.sweep._row_sia``.
    """
    substrate = examples.basic_substrate()
    section = landscape_section(
        lambda _theta: substrate, (1, 0, 0), [0.0], formalism="IIT_3_0"
    )
    row = section.df.iloc[0]
    assert math.isfinite(row["phi"])
    assert isinstance(row["partition"], str)
    for column in (
        "signed_phi",
        "normalized_phi",
        "signed_normalized_phi",
        "partition_margin",
        "cause_state_margin",
        "effect_state_margin",
    ):
        assert math.isnan(row[column])
    assert row["cause_state"] is None
    assert row["effect_state"] is None
    assert row["effectively_tied"] is None
