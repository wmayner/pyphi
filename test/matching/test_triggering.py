"""Tests for triggering coefficients (Eq 5-7)."""

import numpy as np
import pytest

import pyphi
from pyphi.matching.triggered_tpm import build_triggered_tpm
from pyphi.matching.triggering import TriggeringCoefficient
from pyphi.matching.triggering import triggering_coefficient


@pytest.fixture
def relay_ttpm():
    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    return build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )


def test_triggering_coefficient_fully_triggered(relay_ttpm):
    # unit 1 fully determined by stimulus: p=1, q=0.5, c=log2(2)=1, info=1, t=1
    tc = triggering_coefficient(relay_ttpm, (1,), (1,), (1,))
    assert isinstance(tc, TriggeringCoefficient)
    assert tc.p == pytest.approx(1.0)
    assert tc.q == pytest.approx(0.5)
    assert tc.connectedness == pytest.approx(1.0)
    assert tc.value == pytest.approx(1.0)


def test_triggering_coefficient_in_unit_interval(relay_ttpm):
    for state in [(0,), (1,)]:
        for stimulus in [(0,), (1,)]:
            tc = triggering_coefficient(relay_ttpm, (1,), state, stimulus)
            assert 0.0 <= tc.value <= 1.0
            assert tc.connectedness >= 0.0


def test_triggering_coefficient_no_effect_is_zero():
    # system unit ignores the sensory unit: p == q -> connectedness 0 -> t 0
    sbn = np.zeros((2, 2, 2))  # all next-states 0
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    tc = triggering_coefficient(t, (1,), (0,), (1,))
    assert tc.p == pytest.approx(tc.q)
    assert tc.connectedness == pytest.approx(0.0)
    assert tc.value == pytest.approx(0.0)
