"""pyphi.analyze: one high-level entry point for a single system's analysis."""

from __future__ import annotations

import math

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
    try:
        analyze(substrate, state, formalism="IIT_9_0")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown formalism")


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
    assert compact.startswith("Analysis(Φ=")
    assert "\n" not in compact  # one line at LOW


def test_analyze_repr_html_renders():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, state)
    html = result._repr_html_()
    assert "Analysis" in html
