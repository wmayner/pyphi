"""pyphi.analyze: one high-level entry point for a single system's analysis."""

from __future__ import annotations

import math

import pytest

from pyphi import System
from pyphi import examples
from pyphi.analyze import Analysis
from pyphi.analyze import analyze
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.display import LOW


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
    # Under 3.0 the CES is bare Distinctions (no embedded SIA); the bundle
    # still exposes a SIA and a usable phi.
    assert getattr(result.ces, "sia", None) is None
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
        direct = complexes(substrate, (0, 0), SearchBounds())
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
    with pytest.raises(ValueError, match="parallel_kwargs"):
        analyze(substrate, state, parallel_kwargs={})


def test_analyze_grains_rejects_non_bounds():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="must be True or a SearchBounds"):
        analyze(substrate, state, grains=0.5)
    # False is a confusion signal, not "no search".
    with pytest.raises(ValueError, match="must be True or a SearchBounds"):
        analyze(substrate, state, grains=False)


def test_analyze_card_distinguishes_phi_s_from_big_phi():
    """Under IIT 4.0 the card's Φ row is Σφ_d + Σφ_r, not the SIA's φₛ.

    `basic` in state (1, 0, 0) is the case where they diverge visibly: φₛ = 0
    under the 2026 formalism while Φ = 1.0.
    """
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2026):
        result = analyze(substrate, (1, 0, 0))
    rows = {r.label: r.value for sec in result._describe(2).sections for r in sec.rows}
    assert rows["φ_s"] == result.phi == 0.0
    assert rows["Φ"] == result.big_phi == 1.0
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
