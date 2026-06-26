"""Tests for pyphi.macro.tpm: the four-step macro TPM construction."""

import numpy as np
import pytest

from pyphi import exceptions
from pyphi.convert import sbn2sbs
from pyphi.macro.tpm import _background_weights_cause
from pyphi.macro.tpm import _discounted_on_probabilities
from pyphi.macro.tpm import _full_transition_matrix
from pyphi.macro.tpm import _unit_final_state_proportions
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from pyphi.system import System

MIN_TPM = np.array(
    [
        [0.05, 0.05],
        [0.05, 0.06],
        [0.06, 0.05],
        [0.95, 0.95],
    ]
)

CG_TPM = np.array(
    [
        [0.05, 0.05, 0.05, 0.05],
        [0.06, 0.15, 0.05, 0.05],
        [0.15, 0.06, 0.05, 0.05],
        [0.16, 0.16, 0.85, 0.85],
        [0.05, 0.05, 0.06, 0.15],
        [0.06, 0.15, 0.06, 0.15],
        [0.15, 0.06, 0.06, 0.15],
        [0.16, 0.16, 0.86, 0.95],
        [0.05, 0.05, 0.15, 0.06],
        [0.06, 0.15, 0.15, 0.06],
        [0.15, 0.06, 0.15, 0.06],
        [0.16, 0.16, 0.95, 0.86],
        [0.85, 0.85, 0.16, 0.16],
        [0.86, 0.95, 0.16, 0.16],
        [0.95, 0.86, 0.16, 0.16],
        [0.96, 0.96, 0.96, 0.96],
    ]
)


def _bbx_micro_tpm():
    n = 8
    states = [tuple((i >> k) & 1 for k in range(n)) for i in range(2**n)]
    tpm = np.zeros((2**n, n))
    for r, cs in enumerate(states):
        p = tpm[r]
        p[0] = 0.01 + 0.01 * cs[0] + 0.1 * cs[3] + 0.8 * cs[6] + 0.05 * cs[1]
        p[1] = 0.01 + 0.01 * cs[1] + 0.1 * cs[3] + 0.8 * cs[6] + 0.05 * cs[0]
        p[2] = (
            0.01
            + 0.01 * cs[2]
            + 0.85 * int(cs[0] + cs[1] > 0)
            + 0.1 * int(cs[0] + cs[1] == 2)
        )
        p[3] = 0.01 + 0.01 * cs[3] + 0.85 * cs[2] + 0.05 * (cs[0] + cs[1])
        p[4] = 0.01 + 0.01 * cs[4] + 0.1 * cs[7] + 0.8 * cs[2] + 0.05 * cs[5]
        p[5] = 0.01 + 0.01 * cs[5] + 0.1 * cs[7] + 0.8 * cs[2] + 0.05 * cs[4]
        p[6] = (
            0.01
            + 0.01 * cs[6]
            + 0.85 * int(cs[4] + cs[5] > 0)
            + 0.1 * int(cs[4] + cs[5] == 2)
        )
        p[7] = 0.01 + 0.01 * cs[7] + 0.85 * cs[6] + 0.05 * (cs[4] + cs[5])
    return tpm


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


class TestBackgroundWeights:
    def test_cause_weights_hand_bayes(self):
        """q_c (Eq 34) on the asymmetric substrate, system = {0}.

        q_c(w) = sum_{uS} P(earliest | (w, uS)) / sum_u P(earliest | u),
        with earliest = the current state for tau = 1.
        """
        substrate = _asymmetric_substrate()
        earliest = (1, 0, 1, 0)
        q = _background_weights_cause(
            substrate.factored_tpm, system_indices=(0,), earliest=earliest
        )
        on = _flat_on_probabilities(substrate)
        # P(earliest | u) for all 16 prior states u
        likelihood = np.ones(16)
        for i in range(4):
            p = on[:, i]
            likelihood *= p if earliest[i] == 1 else 1 - p
        # background = units 1,2,3; sum over the system bit (unit 0)
        idx = np.arange(16)
        w_index = ((idx >> 1) & 1) | (((idx >> 2) & 1) << 1) | (((idx >> 3) & 1) << 2)
        expected = np.zeros(8)
        np.add.at(expected, w_index, likelihood)
        expected /= likelihood.sum()
        assert np.allclose(q, expected, atol=1e-15)
        assert q.sum() == pytest.approx(1.0)

    def test_unreachable_earliest_state_raises(self):
        # A deterministic substrate where state (0,) cannot be reached:
        tpm = np.array([[1.0], [1.0]])  # always ON
        substrate = Substrate(tpm, node_labels=("A",))
        with pytest.raises(exceptions.StateUnreachableBackwardsError):
            _background_weights_cause(
                substrate.factored_tpm, system_indices=(), earliest=(0,)
            )


class TestHandComputedTinyCase:
    """Authors' 'minimal' example: 2 micro units, 1 macro unit, tau=1.

    alpha = coarse-grain of (A, B), ON iff both ON. Empty background.
    Hand computation (matches the authors' hand-derived min_macro TPM):
      T(alpha'=1 | alpha=0)
        = mean over preimage {00, 10, 01} of p(A'=1)p(B'=1)
        = (0.05*0.05 + 0.05*0.06 + 0.06*0.05) / 3
        = 0.05*0.05 + 2*0.01*0.05/3
      T(alpha'=1 | alpha=1) = 0.95*0.95
    """

    def test_min_macro_tpm(self):
        substrate = Substrate(MIN_TPM, node_labels=("A", "B"))
        units = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        cause, effect = macro_tpms(substrate, units, ((0, 0),))
        expected_off = 0.05 * 0.05 + 2 * 0.01 * 0.05 / 3
        expected_on = 0.95 * 0.95
        for tpm in (cause, effect):
            factor = tpm.factor(0)
            assert factor.shape == (2, 2)
            assert factor[0, 1] == pytest.approx(expected_off, abs=1e-15)
            assert factor[1, 1] == pytest.approx(expected_on, abs=1e-15)
            assert np.allclose(factor.sum(axis=-1), 1.0)


class TestPaperExampleTPMs:
    def test_cg_construction_exact(self):
        """Example 1. The construction values are derived in closed form.

        The authors' committed macro TPM (their repo, results from a
        hand-entered matrix) contains a rounding (0.006833 for 0.0615/9)
        and a hand-entry error (0.9212 where the construction gives
        0.96**2 = 0.9216); rows (1,0)/(0,1) match exactly. The committed
        bbx TPM is computed rather than hand-entered and has no such
        discrepancy (see the next test).
        """
        substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        cause, effect = macro_tpms(substrate, units, ((0, 0, 0, 0),))
        expected = np.array(
            [
                [0.0615 / 9, 0.0615 / 9],
                [0.0256, 0.7855],
                [0.7855, 0.0256],
                [0.9216, 0.9216],
            ]
        )
        for tpm in (cause, effect):
            built = np.stack(
                [tpm.factor(i)[..., 1].reshape(-1, order="F") for i in range(2)],
                axis=1,
            )
            assert np.allclose(built, expected, atol=1e-14)

    def test_bbx_construction_matches_authors_computation(self):
        """Example 2: must equal the authors' computed TPM to ~1e-15.

        Their computation (repo, `_get_blackbox_example_macro_tpm`):
        square the state-by-state micro TPM, condition on the (C, G)
        preimage of the current macro state, and read out C / G at the
        final update. For this wiring that shortcut coincides with the
        full construction because external influence enters each half
        only via the conditioned units at the window start.
        """
        micro = _bbx_micro_tpm()
        substrate = Substrate(micro, node_labels=tuple("ABCDEFGH"))
        units = (
            MacroUnit((0, 1, 2, 3), 2, blackbox(4, 2, (2,))),
            MacroUnit((4, 5, 6, 7), 2, blackbox(4, 2, (2,))),
        )
        ones = (1,) * 8
        cause, effect = macro_tpms(substrate, units, (ones, ones))

        sbs = sbn2sbs(micro)
        tpm2 = sbs @ sbs
        idx = np.arange(2**8)
        c_bit = (idx >> 2) & 1
        g_bit = (idx >> 6) & 1
        expected = np.zeros((4, 2))
        for si, (a, b) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            rows = np.where((c_bit == a) & (g_bit == b))[0]
            expected[si, 0] = tpm2[rows][:, c_bit == 1].sum(axis=1).mean()
            expected[si, 1] = tpm2[rows][:, g_bit == 1].sum(axis=1).mean()

        for tpm in (cause, effect):
            built = np.stack(
                [tpm.factor(i)[..., 1].reshape(-1, order="F") for i in range(2)],
                axis=1,
            )
            assert np.allclose(built, expected, atol=1e-13)


class TestMicroReductionWithBackground:
    """Identity macroing of a proper subset must reproduce System's
    background-conditioned TPMs exactly (Eqs 33-34 reduce to IIT 4.0
    Eq 4 at tau = 1)."""

    @pytest.mark.parametrize("subset", [(0,), (0, 1), (1, 3)])
    def test_identity_subset_equals_proper_tpms(self, subset):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        units = tuple(micro_unit(i) for i in subset)
        cause, effect = macro_tpms(substrate, units, (state,))
        system = System(substrate, state, subset)
        for built, reference in (
            (cause, system.proper_cause_marginal),
            (effect, system.proper_effect_marginal),
        ):
            for k in range(len(subset)):
                assert np.allclose(built.factor(k), reference.factor(k), atol=1e-15)

    def test_cause_and_effect_differ_with_background(self):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        units = (micro_unit(0), micro_unit(1))
        cause, effect = macro_tpms(substrate, units, (state,))
        assert not all(np.allclose(cause.factor(k), effect.factor(k)) for k in range(2))


class TestApportionedBackgroundPath:
    """Eq 29 path: no published anchor (both paper examples have empty
    background) -- unit-level checks only until sub-project 3."""

    def test_apportionment_invisible_at_tau_1(self):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        plain = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        apportioned = (
            MacroUnit(
                (0, 1),
                1,
                coarse_grain(2, on_counts={2}),
                background_apportionment=(3,),
            ),
        )
        # tau = 1: Step 1 noising never feeds back into U^J
        _, effect_plain = macro_tpms(substrate, plain, (state,))
        _, effect_app = macro_tpms(substrate, apportioned, (state,))
        assert np.allclose(effect_plain.factor(0), effect_app.factor(0))

    def test_apportionment_bites_at_tau_2(self):
        # Apportioning only unit 3 would NOT change the TPM here: unit
        # 3's sole input (unit 2) lies outside U union W, so its Eq 29
        # row averages to the same constant as full Eq 28 noise.
        # Apportioning both 2 and 3 keeps unit 3's dependence on unit 2,
        # whose state then feeds back into unit 0 at the second update.
        substrate = _asymmetric_substrate()
        history = ((1, 0, 1, 0), (1, 1, 0, 0))
        plain = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        apportioned = (
            MacroUnit((0, 1), 2, blackbox(2, 2, (0,)), background_apportionment=(2, 3)),
        )
        _, effect_plain = macro_tpms(substrate, plain, history)
        _, effect_app = macro_tpms(substrate, apportioned, history)
        assert not np.allclose(effect_plain.factor(0), effect_app.factor(0))

    def test_outputs_are_stochastic(self):
        substrate = _asymmetric_substrate()
        history = ((1, 0, 1, 0), (1, 1, 0, 0))
        units = (
            MacroUnit((0, 1), 2, blackbox(2, 2, (0,)), background_apportionment=(3,)),
            MacroUnit((2,), 2, blackbox(1, 2, (0,))),
        )
        cause, effect = macro_tpms(substrate, units, history)
        for tpm in (cause, effect):
            for k in range(2):
                assert np.allclose(tpm.factor(k).sum(axis=-1), 1.0)
