"""Tests for pyphi.macro.criteria: intrinsic-unit criteria (Eqs 15-16)."""

import numpy as np
import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import canonical_units
from pyphi.macro.criteria import constituent_system
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.criteria import unit_integration
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from test.macro.test_macro_tpm import MIN_TPM


def min_substrate():
    return Substrate(MIN_TPM, node_labels=("A", "B"))


def bu_substrate():
    """The authors' bottom-up example: 3 units, deterministic TPM.

    State (0, 0, 0). Note: the committed result set for this example was
    generated under old pyphi's SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI
    default (false), which contradicts the authors' committed config and
    their other result sets; the values asserted in this suite are the
    ones the consistent (flag-true) convention produces, verified
    against the authors' pinned pyphi revision during planning.
    """
    rows = [
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]
    return Substrate(np.array(rows, dtype=float), node_labels=("A", "B", "C"))


# judge_candidate never introspects competitor systems, so opaque
# sentinels stand in for MacroSystem objects in these pure-logic tests.
S1, S2, S3 = object(), object(), object()


class TestJudgeCandidate:
    def test_valid_when_integrated_and_maximal(self):
        verdict = judge_candidate(0.5, [(S1, 0.1), (S2, 0.3)])
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        assert verdict.phi == 0.5
        assert verdict.witness is None
        assert verdict.witness_phi is None
        assert verdict.num_competitors == 2

    def test_not_integrated_when_phi_zero(self):
        verdict = judge_candidate(0.0, [(S1, 0.1)])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED
        assert verdict.witness is None
        assert verdict.num_competitors == 1

    def test_not_integrated_at_precision(self):
        # Positive but below precision: not strictly greater than zero.
        verdict = judge_candidate(1e-15, [])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED

    def test_not_maximal_carries_strongest_witness(self):
        verdict = judge_candidate(0.2, [(S1, 0.1), (S2, 0.7), (S3, 0.3)])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_MAXIMAL
        assert verdict.witness is S2
        assert verdict.witness_phi == 0.7

    def test_tied_at_precision(self):
        verdict = judge_candidate(0.5, [(S1, 0.5 + 1e-15)])
        assert not verdict.valid
        assert verdict.reason is Reason.TIED
        assert verdict.witness is S1
        assert verdict.witness_phi == 0.5 + 1e-15

    def test_exact_tie(self):
        verdict = judge_candidate(0.5, [(S1, 0.5)])
        assert not verdict.valid
        assert verdict.reason is Reason.TIED

    def test_no_competitors_valid_iff_integrated(self):
        assert judge_candidate(0.5, []).valid
        assert not judge_candidate(0.0, []).valid

    def test_first_of_equal_witnesses_kept(self):
        verdict = judge_candidate(0.2, [(S1, 0.7), (S2, 0.7)])
        assert verdict.witness is S1

    def test_verdict_is_frozen(self):
        verdict = judge_candidate(0.5, [])
        with pytest.raises(AttributeError):
            verdict.valid = False


class TestConstituentSystem:
    def test_micro_indices_become_micro_units(self):
        system = constituent_system(min_substrate(), (0, 1), ((0, 0),))
        assert system.units == (micro_unit(0), micro_unit(1))

    def test_constituent_order_is_canonical(self):
        a = constituent_system(min_substrate(), (1, 0), ((0, 0),))
        b = constituent_system(min_substrate(), (0, 1), ((0, 0),))
        assert a == b

    def test_bare_state_accepted_at_grain_one(self):
        system = constituent_system(min_substrate(), (0, 1), (0, 0))
        assert system.micro_history == ((0, 0),)

    def test_meso_constituent_keeps_full_definition(self):
        meso = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))
        system = constituent_system(min_substrate(), (meso,), ((0, 0),))
        assert system.units == (meso,)

    def test_history_trimmed_to_constituent_grain(self):
        # Grain-1 constituents with a length-2 history (as supplied for
        # a grain-2 candidate): only the trailing state is used.
        system = constituent_system(min_substrate(), (0, 1), ((1, 1), (0, 0)))
        assert system.micro_history == ((0, 0),)

    def test_history_too_short_rejected(self):
        meso = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        with pytest.raises(ValueError, match="history"):
            constituent_system(min_substrate(), (meso,), ((0, 0),))


class TestUnitIntegration:
    def test_min_pair_anchor(self):
        with config.override(**presets.iit4_2023):
            phi = unit_integration(min_substrate(), (0, 1), ((0, 0),))
            assert phi == pytest.approx(0.005106576483955726, abs=1e-13)

    def test_min_singletons_zero(self):
        with config.override(**presets.iit4_2023):
            assert unit_integration(min_substrate(), (0,), ((0, 0),)) == 0.0
            assert unit_integration(min_substrate(), (1,), ((0, 0),)) == 0.0

    def test_unreachable_state_gives_zero(self):
        # bu unit C is forced ON when (A, B) = (0, 0): the one-unit
        # system over C cannot exist in state 0, so phi_s is zero.
        with config.override(**presets.iit4_2023):
            assert unit_integration(bu_substrate(), (2,), ((0, 0, 0),)) == 0.0

    def test_bu_singleton_anchors(self):
        # See bu_substrate's docstring for why these are 1.0, not the
        # stale committed 0.0.
        with config.override(**presets.iit4_2023):
            assert unit_integration(bu_substrate(), (0,), ((0, 0, 0),)) == 1.0
            assert unit_integration(bu_substrate(), (1,), ((0, 0, 0),)) == 1.0


class TestCanonicalUnits:
    def test_sorted_by_footprint(self):
        units = canonical_units([micro_unit(1), micro_unit(0)])
        assert units == (micro_unit(0), micro_unit(1))
