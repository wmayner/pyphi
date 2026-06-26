"""Tests for the Perception view (single-stimulus perception)."""

import numpy as np
import pytest

from pyphi import examples
from pyphi.matching import PerceptualSystem
from pyphi.matching.perception import Perception


def _full_state(sensory_indices, system_indices, x, y):
    n = len(sensory_indices) + len(system_indices)
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, yi in zip(system_indices, y, strict=True):
        full[i] = yi
    return tuple(full)


@pytest.fixture(scope="module")
def perception():
    # grid3 over (1,2) yields 3 distinctions and 5 relations -> exercises the
    # relation-perception path (basic_substrate over (1,2) has no relations).
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    stimulus = (1,)
    y = ttpm.argmax_state(stimulus)
    ces = substrate.ces(state=_full_state(sensory, system, stimulus, y), indices=system)
    return Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)


def test_distinction_perception_is_t_times_phi(perception):
    for d in perception.ces.distinctions:
        tc = perception.triggering_coefficients[d.mechanism]
        assert perception.distinction_perception(d) == pytest.approx(
            tc.value * float(d.phi)
        )


def test_distinction_perception_at_most_phi(perception):
    for d in perception.ces.distinctions:
        assert perception.distinction_perception(d) <= float(d.phi) + 1e-12


def test_relation_perception_is_phi_times_mean_t(perception):
    for r in perception.ces.relations:
        mean_t = np.mean(
            [perception.triggering_coefficients[rel.mechanism].value for rel in r]
        )
        assert perception.relation_perception(r) == pytest.approx(float(r.phi) * mean_t)


def test_richness_is_sum_of_component_perceptions(perception):
    expected = sum(
        perception.distinction_perception(d) for d in perception.ces.distinctions
    ) + sum(perception.relation_perception(r) for r in perception.ces.relations)
    assert perception.richness == pytest.approx(expected)


def test_fold_perception_uses_big_phi_contribution(perception):
    d = next(iter(perception.ces.distinctions))
    fold = perception.ces.fold([d])
    tc = perception.triggering_coefficients[d.mechanism]
    assert perception.fold_perception(fold) == pytest.approx(
        tc.value * fold.big_phi_contribution
    )


def test_consistency_guard_rejects_wrong_state():
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    stimulus = (1,)
    y = ttpm.argmax_state(stimulus)
    wrong = tuple(1 - v for v in y)
    ces = substrate.ces(
        state=_full_state(sensory, system, stimulus, wrong), indices=system
    )
    with pytest.raises(ValueError, match="triggered"):
        Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)
