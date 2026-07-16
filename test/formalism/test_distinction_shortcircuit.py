"""Tests for distinction-level reducibility short-circuiting
(``formalism.iit.shortcircuit_distinctions``)."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi import numerics
from pyphi.conf.formalism import IITConfig
from pyphi.direction import Direction
from pyphi.formalism import queries
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import _null_ria
from pyphi.models.explanation import NullResultReason
from pyphi.substrate import Substrate
from pyphi.system import System
from test.conftest import IIT_3_CONFIG
from test.conftest import IIT_4_CONFIG


@pytest.fixture(autouse=True)
def _pin_formalism():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield


def test_shortcircuit_distinctions_default_true():
    assert IITConfig().shortcircuit_distinctions is True


def test_shortcircuit_distinctions_must_be_bool():
    with pytest.raises(ValueError, match="shortcircuit_distinctions"):
        IITConfig(shortcircuit_distinctions="yes")


def test_presets_carry_shortcircuit_distinctions():
    from pyphi.conf import presets

    for preset in (presets.iit3, presets.iit4_2023, presets.iit4_2026):
        assert preset["iit"].shortcircuit_distinctions is True


@pytest.fixture
def sink_system():
    """A → B → C chain; C has no outputs, so a mechanism containing only C
    has an empty candidate effect-purview set. A takes no inputs and is
    always 0; B copies A; C copies B."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    # fmt: on
    substrate = Substrate(tpm, cm=cm, node_labels=("A", "B", "C"))
    return System(substrate, (0, 0, 0))


def _recording_find_mice(monkeypatch):
    """Wrap queries.find_mice to record the directions it is called with."""
    calls = []
    real = queries.find_mice

    def recording(cs, direction, mechanism, **kwargs):
        calls.append(direction)
        return real(cs, direction, mechanism, **kwargs)

    monkeypatch.setattr(queries, "find_mice", recording)
    return calls


def test_cause_search_skipped_when_effect_trivially_reducible(sink_system, monkeypatch):
    calls = _recording_find_mice(monkeypatch)
    d = queries.distinction(sink_system, (2,))
    assert Direction.CAUSE not in calls
    assert not numerics.is_positive(d.phi)
    assert tuple(d.effect.reasons) == (NullResultReason.NO_PURVIEWS,)
    assert tuple(d.cause.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_effect_search_skipped_when_cause_phi_zero(monkeypatch):
    calls = []
    real = queries.find_mice

    def zero_cause(cs, direction, mechanism, **kwargs):
        calls.append(direction)
        if direction == Direction.CAUSE:
            return MaximallyIrreducibleCause(
                _null_ria(
                    Direction.CAUSE,
                    mechanism,
                    (0,),
                    reasons=(NullResultReason.REDUCIBLE_OVER_PARTITION,),
                )
            )
        return real(cs, direction, mechanism, **kwargs)

    monkeypatch.setattr(queries, "find_mice", zero_cause)
    system = examples.basic_system()
    d = queries.distinction(system, (0,))
    assert calls == [Direction.CAUSE]
    assert not numerics.is_positive(d.phi)
    assert tuple(d.effect.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_flag_off_evaluates_both_directions(sink_system, monkeypatch):
    calls = _recording_find_mice(monkeypatch)
    with pyphi.config.override(shortcircuit_distinctions=False):
        d = queries.distinction(sink_system, (2,))
    assert Direction.CAUSE in calls
    assert Direction.EFFECT in calls
    assert NullResultReason.OTHER_DIRECTION_REDUCIBLE not in (d.cause.reasons or ())
    assert tuple(d.effect.reasons) == (NullResultReason.NO_PURVIEWS,)


def test_ces_identical_with_and_without_shortcircuit():
    system = examples.basic_system()
    with pyphi.config.override(shortcircuit_distinctions=True):
        on = list(system.all_distinctions())
    with pyphi.config.override(shortcircuit_distinctions=False):
        off = list(system.all_distinctions())
    assert on == off


def test_iit3_concept_shortcircuits(sink_system, monkeypatch):
    calls = _recording_find_mice(monkeypatch)
    from pyphi.formalism import iit3

    with IIT_3_CONFIG, pyphi.config.override(progress_bars=False):
        c = iit3.concept(sink_system, (2,))
    assert Direction.CAUSE not in calls
    assert tuple(c.cause.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_iit3_sia_unchanged_by_shortcircuit():
    """Confirmation experiment for the spec's verification point: the IIT 3.0
    partitioned-constellation path consumes nothing from skipped MICEs, so the
    SIA is identical with the flag on and off."""
    with IIT_3_CONFIG, pyphi.config.override(progress_bars=False):
        system = examples.basic_system()
        with pyphi.config.override(shortcircuit_distinctions=True):
            sia_on = system.sia()
        with pyphi.config.override(shortcircuit_distinctions=False):
            sia_off = system.sia()
    assert numerics.eq(sia_on.phi, sia_off.phi)
    assert sia_on.partition == sia_off.partition


def test_phi_skips_effect_mip_when_cause_phi_zero(sink_system, monkeypatch):
    calls = []
    real = queries.phi_effect_mip

    def recording(cs, mechanism, purview, **kwargs):
        calls.append((mechanism, purview))
        return real(cs, mechanism, purview, **kwargs)

    monkeypatch.setattr(queries, "phi_effect_mip", recording)
    # Purview (0,) is not a potential cause purview of mechanism (2,)
    # (A receives no edge from C), so the cause MIP is null with φ = 0.
    result = queries.phi(sink_system, (2,), (0,))
    assert result == 0
    assert calls == []
