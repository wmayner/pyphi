"""Tests for pyphi.macro.tpm: the four-step macro TPM construction."""

import numpy as np
import pytest

from pyphi.macro.tpm import _discounted_on_probabilities
from pyphi.macro.tpm import _full_transition_matrix
from pyphi.macro.tpm import _unit_final_state_proportions
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.substrate import Substrate


def _asymmetric_substrate():
    """Asymmetric 4-unit substrate: every unit has a distinct rule, so axis
    or endianness errors cannot cancel."""
    n = 4
    states = [tuple((i >> k) & 1 for k in range(n)) for i in range(2**n)]
    tpm = np.zeros((2**n, n))
    for r, s in enumerate(states):
        tpm[r, 0] = 0.1 + 0.5 * s[1] + 0.3 * s[3]
        tpm[r, 1] = 0.2 + 0.7 * s[0]
        tpm[r, 2] = 0.05 + 0.6 * s[0] * s[1] + 0.3 * s[3]
        tpm[r, 3] = 0.9 - 0.8 * s[2]
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


def _flat_on_probabilities(substrate):
    """(2**n, n) ON probabilities, little-endian rows, from the factors."""
    factored = substrate.factored_tpm
    n = factored.n_nodes
    return np.stack(
        [factored.factor(i)[..., 1].reshape(-1, order="F") for i in range(n)],
        axis=1,
    )


class TestDiscounting:
    def test_constituent_rows_untouched(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm, units, 0)
        # Eq 27: units 0 and 1 keep all connections
        assert np.array_equal(discounted[:, 0], original[:, 0])
        assert np.array_equal(discounted[:, 1], original[:, 1])

    def test_other_system_units_fully_noised(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm, units, 0)
        # Eq 28: units 2 and 3 become the uniform-average marginal
        for i in (2, 3):
            assert np.allclose(discounted[:, i], original[:, i].mean())

    def test_unapportioned_background_fully_noised(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm, units, 0)
        for i in (2, 3):
            assert np.allclose(discounted[:, i], original[:, i].mean())

    def test_apportioned_background_keeps_patron_inputs_only(self):
        substrate = _asymmetric_substrate()
        # Unit over (0, 1) with background unit 2 apportioned to it;
        # unit 3 is unapportioned background. Unit 2's original rule is
        # 0.05 + 0.6*s0*s1 + 0.3*s3: after Eq 29 noising over unit 3
        # (outside U union W = {0, 1, 2}) the row keeps its dependence
        # on the patron constituents 0, 1 but averages s3 to 0.5.
        units = (
            MacroUnit(
                (0, 1),
                1,
                coarse_grain(2, on_counts={2}),
                background_apportionment=(2,),
            ),
        )
        discounted = _discounted_on_probabilities(substrate.factored_tpm, units, 0)
        n = 4
        idx = np.arange(2**n)
        s0 = (idx >> 0) & 1
        s1 = (idx >> 1) & 1
        expected = 0.05 + 0.6 * s0 * s1 + 0.3 * 0.5
        assert np.allclose(discounted[:, 2], expected, atol=1e-15)
        # Unit 3 is unapportioned: fully noised (Eq 28)
        original = _flat_on_probabilities(substrate)
        assert np.allclose(discounted[:, 3], original[:, 3].mean())

    def test_little_endian_row_order_pinned(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1, 2, 3), 1, coarse_grain(4, on_counts={4})),)
        discounted = _discounted_on_probabilities(substrate.factored_tpm, units, 0)
        # row index 1 = state (1,0,0,0): unit 1's rule = 0.2 + 0.7*s[0]
        assert discounted[1, 1] == pytest.approx(0.9)
        # row index 2 = state (0,1,0,0): unit 0's rule = 0.1 + 0.5*s[1]
        assert discounted[2, 0] == pytest.approx(0.6)


class TestTransitionMatrix:
    def test_rows_stochastic(self):
        substrate = _asymmetric_substrate()
        on = _flat_on_probabilities(substrate)
        P = _full_transition_matrix(on)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_hand_checked_entry(self):
        substrate = _asymmetric_substrate()
        on = _flat_on_probabilities(substrate)
        P = _full_transition_matrix(on)
        # From state (1,0,0,0) (row 1) to state (1,1,0,0) (column 3):
        # at s = (1,0,0,0): pA = 0.1, pB = 0.9, pC = 0.05, pD = 0.9
        # p = pA * pB * (1 - pC) * (1 - pD)
        expected = 0.1 * 0.9 * 0.95 * 0.1
        assert P[1, 3] == pytest.approx(expected, abs=1e-15)


class TestFinalStateProportions:
    def test_tau1_uniform_over_preimage(self):
        unit = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))
        r0 = _unit_final_state_proportions(unit, 0)
        r1 = _unit_final_state_proportions(unit, 1)
        assert np.allclose(r0, [1 / 3, 1 / 3, 1 / 3, 0.0])
        assert np.allclose(r1, [0.0, 0.0, 0.0, 1.0])

    def test_blackbox_counts_prefix_multiplicity(self):
        # tau = 2 over 1 constituent, state = final update: preimages
        # {(a, j) : a free} -> uniform over the final state only.
        unit = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        assert np.allclose(_unit_final_state_proportions(unit, 1), [0.0, 1.0])
        assert np.allclose(_unit_final_state_proportions(unit, 0), [1.0, 0.0])

    def test_sums_to_one(self):
        unit = MacroUnit((0, 1), 2, blackbox(2, 2, (1,)))
        for j in (0, 1):
            assert _unit_final_state_proportions(unit, j).sum() == pytest.approx(1.0)
