"""pyphi.analyze: one high-level entry point for a single system's analysis."""

from __future__ import annotations

import math

import pytest

from pyphi import System
from pyphi import examples
from pyphi.analysis import Analysis
from pyphi.analysis import analyze
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.conf.formalism import IITConfig
from pyphi.display import LOW
from pyphi.formalism import iit4
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.distinctions import UnresolvedDistinctions
from test.conftest import skip_if_no_emd_backend
from test.hypothesis_utils import random_substrate


def test_analyze_bundle_parity_with_substrate_sia():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
        direct = substrate.sia(state).phi
    assert isinstance(result, Analysis)
    assert math.isclose(result.phi, float(direct))


def test_analyze_bundle_embeds_sia_under_iit4():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
    # Under 4.0 the CES embeds the SIA; the bundle reuses it (phi-equal).
    assert result.ces.sia is result.sia
    assert math.isclose(result.phi, float(result.ces.sia.phi))


def test_analyze_compute_sia_returns_raw_object():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state, compute="sia")
        direct = System(substrate, state).sia()
    assert not isinstance(result, Analysis)
    assert math.isclose(float(result.phi), float(direct.phi))


def test_analyze_compute_ces_returns_raw_object():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state, compute="ces")
        direct = System(substrate, state).ces()
    assert not isinstance(result, Analysis)
    assert math.isclose(float(result.sia.phi), float(direct.sia.phi))


def test_analyze_compute_callable():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state, compute=lambda system: len(system))
    assert result == len(substrate)


def test_analyze_inline_formalism_restores_global_config():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    before = config.formalism.iit.version
    analyze(substrate, state, formalism="IIT_4_0_2023")
    assert config.formalism.iit.version == before


def test_analyze_iit3_bundle_pairs_distinctions_and_sia():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    result = analyze(substrate, state, formalism="IIT_3_0")
    assert isinstance(result, Analysis)
    # Under 3.0, as under 4.0, the CES embeds the SIA the bundle exposes.
    assert result.ces.sia is result.sia
    assert isinstance(result.phi, float)


def test_analyze_subset_analyzes_subsystem():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state, subset=(0, 1))
    assert len(result.system) == 2


def test_analyze_unknown_formalism_raises_valueerror():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError):
        analyze(substrate, state, formalism="IIT_9_0")


def test_analyze_repr_renders_full_card():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
    text = repr(result)
    assert "Analysis" in text
    assert "Distinctions" in text  # the flat CES card is folded in


def test_analyze_repr_compact_at_low_verbosity():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
        with config.override(repr_verbosity=LOW):
            compact = repr(result)
    assert compact.startswith("Analysis(φ_s=")
    assert "\n" not in compact  # one line at LOW


def test_analyze_repr_html_renders():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
    html = result._repr_html_()
    assert "Analysis" in html


def test_analyze_to_pandas_one_row_columns():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
    df = result.to_pandas()
    assert len(df) == 1
    assert set(df.columns) == {
        "formalism",
        "subset",
        "state",
        "current_state",
        "phi",
        "normalized_phi",
        "big_phi",
        "n_distinctions",
        "sum_phi_d",
        "sum_phi_r",
    }
    assert math.isclose(float(df.iloc[0]["phi"]), result.phi)


def test_analyze_exported_at_package_root():
    import pyphi

    assert pyphi.analyze is analyze
    assert pyphi.Analysis is Analysis


def test_analyze_unknown_compute_string_raises_valueerror():
    """A compute-string typo must raise, not silently run the full bundle."""
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="SIA"):
        analyze(substrate, state, formalism="IIT_4_0_2023", compute="SIA")


def test_analyze_grains_matches_macro_complexes():
    from pyphi.macro.search import ComplexesResult
    from pyphi.macro.search import SearchBounds
    from pyphi.macro.search import complexes
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with config.override(**presets.iit4_2023):
        via_analyze = analyze(substrate, (0, 0), grains=True)
        direct = complexes(substrate, (0, 0), bounds=SearchBounds())
    assert isinstance(via_analyze, ComplexesResult)
    assert via_analyze.maximal_complex.units == direct.maximal_complex.units
    assert float(via_analyze.maximal_complex.phi) == pytest.approx(
        float(direct.maximal_complex.phi), abs=1e-13
    )
    assert len(via_analyze.records) == len(direct.records)


def test_analyze_grains_accepts_bounds_instance():
    from pyphi.macro.search import SearchBounds
    from pyphi.models.complex import Complex
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, (0, 0), grains=SearchBounds(max_depth=0))
    assert len(result.complexes) == 1
    assert all(isinstance(c, Complex) for c in result.complexes)


def test_analyze_grains_iit3_raises():
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with pytest.raises(ValueError, match="IIT_3_0"):
        analyze(substrate, (0, 0), formalism="IIT_3_0", grains=True)


def test_analyze_grains_mutual_exclusions():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="subset"):
        analyze(substrate, state, grains=True, subset=(0,))
    with pytest.raises(ValueError, match="compute"):
        analyze(substrate, state, grains=True, compute="sia")


def test_analyze_grains_rejects_non_bounds():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="must be True, False, or a SearchBounds"):
        analyze(substrate, state, grains=0.5)


def test_analyze_card_distinguishes_phi_s_from_big_phi():
    """Under IIT 4.0 the card's Φ row is Σφ_d + Σφ_r, not the SIA's φₛ.

    `basic` in state (1, 0, 0) is the case where they diverge visibly: φₛ = 0
    under the 2026 formalism while Φ = 1.125.
    """
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        result = analyze(substrate, (1, 0, 0))
    rows = {r.label: r.value for sec in result._describe(2).sections for r in sec.rows}
    assert rows["φ_s"] == result.phi == 0.0
    assert rows["Φ"] == result.big_phi == 1.125
    assert rows["Φ"] == rows["Σφ_d"] + rows["Σφ_r"]


def test_analyze_big_phi_undefined_under_iit3():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit3):
        result = analyze(substrate, state)
        with pytest.raises(AttributeError, match="no structure integrated"):
            _ = result.big_phi
        # IIT 3.0's system-level value *is* its Φ, so the card says so.
        with config.override(repr_verbosity=LOW):
            assert repr(result).startswith("Analysis(Φ=")


def test_analyze_compute_distinctions_matches_the_ces_when_untied():
    """A cause-effect structure filters its distinctions for congruence with
    the specified state the SIA carries. When that state is untied, the SIA
    leaves it exactly as the system's intrinsic information produced it, so
    the distinctions can be filtered without any system partition — and the
    result must be identical."""
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        result = analyze(substrate, (1, 0, 0), compute="distinctions")
        ces = analyze(substrate, (1, 0, 0), compute="ces")
    assert isinstance(result, ResolvedDistinctions)
    assert result == ces.distinctions


def test_analyze_compute_distinctions_leaves_a_tied_state_unresolved():
    """A tie in the specified state is broken by the φ_s cascade over the
    tied cause/effect pairs, which needs the partition search; the cheap path
    must return the unfiltered distinctions rather than guess."""
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        result = analyze(substrate, (1, 1, 0), compute="distinctions")
    assert isinstance(result, UnresolvedDistinctions)


def test_analyze_compute_distinctions_is_resolved_under_iit3():
    """IIT 3.0 has no tied specified states and no congruence filter, so its
    distinctions are the cause-effect structure's as computed."""
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit3):
        result = analyze(substrate, state, compute="distinctions")
        ces = analyze(substrate, state, compute="ces")
    assert isinstance(result, ResolvedDistinctions)
    assert result == ces.distinctions


def test_analyze_compute_distinctions_enumerates_no_system_partition(monkeypatch):
    """The saving is the partition sweep itself, which for a sparse substrate
    is the whole cost of a cause-effect structure."""
    from pyphi import partition

    calls = []
    original = partition.system_partitions

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(partition, "system_partitions", spy)
    monkeypatch.setattr(iit4, "system_partitions", spy)
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        analyze(substrate, (1, 0, 0), compute="distinctions")
        assert not calls
        analyze(substrate, (1, 0, 0), compute="ces")
    assert calls


def test_analysis_reports_its_formalism():
    """A result records which formalism produced it. The card shows the
    version only when it is not the default."""
    substrate = examples.iit4_2023_fig1a_substrate()
    for name in ("IIT_4_0_2026", "IIT_4_0_2023", "IIT_3_0"):
        with config.override(**presets.by_name[name], progress_bars=False):
            analysis = analyze(substrate, (0, 1, 1), subset=(0, 1))
        assert analysis.formalism == name
        card = repr(analysis)
        if name == IITConfig.version:
            assert "Formalism" not in card
            assert name not in card
        else:
            assert "Formalism" in card
            assert name in card


def test_analyze_to_pandas_identifies_the_analysis():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        result = analyze(substrate, (1, 0, 0), subset=(1, 0))
    row = result.to_pandas().iloc[0]
    sia_row = result.sia.to_pandas().iloc[0]
    assert row["formalism"] == "IIT_4_0_2026"
    assert row["subset"] == sia_row["subset"] == (0, 1)
    assert row["current_state"] == sia_row["current_state"]


FORMALISM_ALIASES = {
    "IIT_4_0_2026": "iit4_2026",
    "IIT_4_0_2023": "iit4_2023",
    "IIT_3_0": "iit3",
}


@pytest.mark.parametrize(
    "name",
    [
        "IIT_4_0_2026",
        "IIT_4_0_2023",
        pytest.param("IIT_3_0", marks=skip_if_no_emd_backend),
    ],
)
def test_every_formalism_spelling_selects_the_same_analysis(name):
    """formalism= as a version name, in lower case, as the preset's exported
    name or as the preset itself gives the analysis that
    config.override(**preset) gives, whichever formalism is active, and
    leaves the active configuration in place."""
    substrate, state = random_substrate(52, 3), (0, 0, 1)
    preset = presets.by_name[name]
    with config.override(**preset):
        expected = analyze(substrate, state, parallel=False)
        expected_distinctions = analyze(
            substrate, state, compute="distinctions", parallel=False
        )
        expected_complexes = substrate.complexes(state, parallel=False)
    assert expected.phi > 0 and expected.ces.distinctions
    alias = FORMALISM_ALIASES[name]
    active = next(other for other in presets.by_name if other != name)
    with config.override(**presets.by_name[active]):
        for spelled in (name, name.lower(), alias, alias.upper(), preset, dict(preset)):
            got = analyze(substrate, state, formalism=spelled, parallel=False)
            assert got.formalism == name
            assert got == expected
        system = System(substrate, state)
        assert system.sia(formalism=alias) == expected.sia
        assert system.ces(formalism=alias) == expected.ces
        assert (
            analyze(
                substrate,
                state,
                formalism=alias,
                compute="distinctions",
                parallel=False,
            )
            == expected_distinctions
        )
        assert substrate.complexes(state, formalism=alias) == expected_complexes
        assert config.formalism.iit.version == active


COMPUTE_CASES = [
    pytest.param("IIT_4_0_2026", {"seed": 52, "n": 3}, (0, 0, 1), None),
    pytest.param(
        "IIT_4_0_2026",
        {"seed": 18, "n": 3, "alphabet": (3, 2, 2)},
        (1, 1, 1),
        None,
    ),
    pytest.param(
        "IIT_4_0_2023", {"seed": 81, "n": 4, "extreme": 0.3}, (1, 0, 1, 0), (0, 1, 2)
    ),
    pytest.param(
        "IIT_3_0",
        {"seed": 52, "n": 3},
        (0, 0, 1),
        None,
        marks=skip_if_no_emd_backend,
    ),
]


@pytest.mark.parametrize(("name", "kwargs", "state", "subset"), COMPUTE_CASES)
def test_compute_modes_agree_with_the_full_analysis(name, kwargs, state, subset):
    """compute="sia", "ces" and "distinctions" return the parts of the full
    analysis they share."""
    substrate = random_substrate(**kwargs)
    with config.override(**presets.by_name[name]):
        full = analyze(substrate, state, subset=subset, parallel=False)
        sia = analyze(substrate, state, subset=subset, compute="sia", parallel=False)
        ces = analyze(substrate, state, subset=subset, compute="ces", parallel=False)
        distinctions = analyze(
            substrate, state, subset=subset, compute="distinctions", parallel=False
        )
    assert full.phi > 0 and full.ces.distinctions
    assert sia == full.sia
    assert ces == full.ces
    assert isinstance(distinctions, ResolvedDistinctions)
    assert distinctions == full.ces.distinctions
