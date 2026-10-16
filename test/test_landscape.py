"""Tests for pyphi.landscape: parameter sections and local derivatives."""

import math

import numpy as np
import pandas as pd
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
    assert list(section.df["theta"]) == grid
    assert isinstance(section.df.index, pd.RangeIndex)
    for _, row in section.df.iterrows():
        theta = row["theta"]
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
    ``pyphi.sweeps._row_sia``.
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


def test_an_empty_grid_raises():
    substrate = examples.basic_substrate()
    with pytest.raises(ValueError, match="grid is empty"):
        landscape_section(lambda _theta: substrate, (1, 0, 0), [])


@pytest.mark.parametrize("suffix", [".json", ".msgpack"])
def test_section_and_perturbation_round_trip(axis, tmp_path, suffix):
    with pyphi.config.override(**pyphi.iit4_2026):
        section = landscape_section(axis, STATE, [0.55, 0.7])
        perturbation = perturb(axis, STATE, 0.7)
    path = tmp_path / f"section{suffix}"
    pyphi.save(section, path)
    assert pyphi.load(path) == section
    path = tmp_path / f"perturbation{suffix}"
    pyphi.save(perturbation, path)
    loaded = pyphi.load(path)
    pd.testing.assert_frame_equal(loaded.to_pandas(), perturbation.to_pandas())
    assert loaded.sias == perturbation.sias
    assert loaded == perturbation


def test_perturbation_with_an_undefined_margin_round_trips(axis, tmp_path):
    """An undefined margin derivative is NaN, and NaN must compare equal to
    itself for a loaded Perturbation to equal the original."""
    with pyphi.config.override(**pyphi.iit4_2026):
        perturbation = perturb(axis, STATE, 1.1)
    assert any(math.isnan(v) for v in perturbation.margin_derivatives.values())
    path = tmp_path / "perturbation.json"
    pyphi.save(perturbation, path)
    assert pyphi.load(path) == perturbation


def test_single_results_are_one_row_frames(axis):
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(axis(0.7), STATE, compute="sia")
        perturbation = perturb(axis, STATE, 0.7)
    for frame in (sia.to_pandas(), perturbation.to_pandas()):
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 1
    record = sia.to_pandas().iloc[0]
    assert record["subset"] == (0, 1, 2)
    assert record["phi"] == pytest.approx(float(sia.phi))


def test_optimization_result_card_is_compact():
    from pyphi.optimization import weight_axes

    builder = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.optimize(builder, STATE, [(-1, 1)], seed=0, maxiter=2, popsize=3)
    text = repr(result)
    assert text.count("\n") < 15
    assert "OptimizationResult" in text
    assert "Substrate" not in text


def test_landscape_partition_column_matches_the_sia_frame():
    with pyphi.config.override(**pyphi.iit4_2026):
        substrate = examples.iit4_2023_fig1a_substrate()
        section = landscape_section(lambda _theta: substrate, (0, 1, 1), [0.0])
        sia = pyphi.analyze(substrate, (0, 1, 1), compute="sia")
    assert section.df["partition"].iloc[0] == sia.to_pandas()["partition"].iloc[0]


def _rc_part_id(partition):
    """The partition string release candidates 1-3 wrote."""
    if partition is None or getattr(partition, "is_null", False):
        return "NullCut"
    set_partition = getattr(partition, "set_partition", None)
    if set_partition is not None:
        parts = str(sorted(sorted(part) for part in set_partition))
    else:
        parts = type(partition).__name__
    return f"{parts}|{sorted(partition.removed_edges())}"


@pytest.mark.parametrize(
    ("formalism", "subset", "state"),
    [
        ("IIT_4_0_2026", None, (0, 1, 1)),
        ("IIT_4_0_2026", (0, 2), (1, 0, 0)),
        pytest.param("IIT_3_0", None, (1, 0, 0), marks=skip_if_no_emd_backend),
    ],
)
def test_a_section_saved_by_a_release_candidate_loads_with_current_partitions(
    tmp_path, formalism, subset, state
):
    """rc1-rc3 wrote '[[0, 1], [2]]|[(2, 0), (2, 1)]'; loading gives the string
    SystemIrreducibilityAnalysis.to_pandas() writes."""
    from dataclasses import replace

    substrate = examples.iit4_2023_fig1a_substrate()
    with pyphi.config.override(**pyphi.conf.presets.by_name[formalism]):
        section = landscape_section(
            lambda _theta: substrate, state, [0.0, 1.0], subset=subset
        )
    old = section.df.copy()
    old["partition"] = [_rc_part_id(sia.partition) for sia in section.sias]
    assert (old["partition"] != section.df["partition"]).all()
    path = tmp_path / "section.json"
    pyphi.save(replace(section, df=old), path)
    pd.testing.assert_frame_equal(pyphi.load(path).df, section.df)
