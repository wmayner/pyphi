"""Tests for distinction-level reducibility short-circuiting
(``formalism.iit.shortcircuit_distinctions``)."""

import pytest

import pyphi
from pyphi.conf.formalism import IITConfig
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
