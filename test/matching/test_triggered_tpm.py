"""Tests for the triggered TPM and its clamp-then-noise construction."""

import numpy as np
import pytest

from pyphi import examples
from pyphi import utils
from pyphi.matching.triggered_tpm import build_triggered_tpm


@pytest.fixture(scope="module")
def ttpm():
    substrate = examples.basic_substrate()  # 3 binary units A,B,C
    return build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=2, tau_clamp=1
    )


def test_array_shape_is_sensory_then_system_axes(ttpm):
    # 1 sensory unit + 2 system units -> shape (2, 2, 2)
    assert ttpm.array.shape == (2, 2, 2)


def test_rows_are_distributions(ttpm):
    for x in utils.all_states(1):
        row = ttpm.row(x)
        assert row.sum() == pytest.approx(1.0)
        assert np.all(row >= 0)


def test_argmax_state_in_support(ttpm):
    for x in utils.all_states(1):
        state = ttpm.argmax_state(x)
        assert len(state) == 2
        assert ttpm.row(x)[state] > 0


def test_tau_clamp_zero_is_pure_noise():
    substrate = examples.basic_substrate()
    noised = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=2, tau_clamp=0
    )
    rows = [noised.row(x) for x in utils.all_states(1)]
    assert np.allclose(rows[0], rows[1])


def test_to_pandas_round_trips(ttpm):
    df = ttpm.to_pandas()
    assert df.shape == (2, 4)
    for x in utils.all_states(1):
        for s in utils.all_states(2):
            assert df.loc[x, s] == pytest.approx(ttpm.array[x + s])


def test_relay_triggered_tpm_hand_computed():
    import pyphi

    # 2 binary units; unit 1 (system) next-state = current unit 0 (sensory).
    # state-by-node shape (2, 2, 2): [a, b] -> P(node ON) for [unit0, unit1].
    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 0] = 0.0  # unit 0 (sensory) — clamped/marginalized
            sbn[a, b, 1] = a  # unit 1 copies unit 0
    substrate = pyphi.Substrate(sbn)

    ttpm = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    # stimulus 0 -> system unit OFF w.p. 1; stimulus 1 -> ON w.p. 1
    assert ttpm.row((0,))[(0,)] == 1.0
    assert ttpm.row((1,))[(1,)] == 1.0
    assert ttpm.argmax_state((0,)) == (0,)
    assert ttpm.argmax_state((1,)) == (1,)


def test_conditional_probability_relay():
    import pyphi

    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    assert t.conditional_probability((1,), (1,), (1,)) == pytest.approx(1.0)
    assert t.conditional_probability((1,), (0,), (1,)) == pytest.approx(0.0)
    assert t.conditional_probability((1,), (0,), (0,)) == pytest.approx(1.0)


def test_marginal_probability_relay():
    import pyphi

    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    assert t.marginal_probability((1,), (1,)) == pytest.approx(0.5)
    assert t.marginal_probability((1,), (0,)) == pytest.approx(0.5)


def test_conditional_probability_subset_marginalizes(ttpm):
    for x in utils.all_states(1):
        p1 = ttpm.conditional_probability((1,), (1,), x)
        row = ttpm.row(x)  # shape (2, 2): axes = unit1, unit2
        assert p1 == pytest.approx(row[1, :].sum())


def test_marginalization_rejects_out_of_system_mechanism(ttpm):
    with pytest.raises(ValueError):
        ttpm.conditional_probability((0,), (1,), (0,))


def test_system_axes_follow_unit_order():
    import pyphi

    # 3 units; unit 1 copies unit 0, unit 2 is constant OFF. The triggered
    # distribution for stimulus (1,) puts all mass on (unit1, unit2) = (1, 0)
    # — an asymmetric state, so any axis-order mixup is visible.
    sbn = np.zeros((2, 2, 2, 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                sbn[a, b, c, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=1, tau_clamp=1
    )
    assert t.argmax_state((1,)) == (1, 0)
    assert t.conditional_probability((1,), (1,), (1,)) == pytest.approx(1.0)
    assert t.conditional_probability((2,), (1,), (1,)) == pytest.approx(0.0)
    assert t.row((1,))[1, 0] == pytest.approx(1.0)
    assert t.array[1, 1, 0] == pytest.approx(1.0)
    df = t.to_pandas()
    assert df.loc[(1,), (1, 0)] == pytest.approx(1.0)


def test_sensory_axes_follow_unit_order():
    import pyphi

    # 3 units; unit 2 (system) copies unit 0 (the first sensory unit) only,
    # so the rows for stimuli (1, 0) and (0, 1) differ.
    sbn = np.zeros((2, 2, 2, 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                sbn[a, b, c, 2] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0, 1), system_indices=(2,), tau=1, tau_clamp=1
    )
    assert t.conditional_probability((2,), (1,), (1, 0)) == pytest.approx(1.0)
    assert t.conditional_probability((2,), (1,), (0, 1)) == pytest.approx(0.0)
    assert t.row((1, 0))[(1,)] == pytest.approx(1.0)
    assert t.row((0, 1))[(0,)] == pytest.approx(1.0)
