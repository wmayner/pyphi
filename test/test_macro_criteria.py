"""Tests for pyphi.macro.criteria: intrinsic-unit criteria (Eqs 15-16)."""

import pytest

from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import judge_candidate

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
