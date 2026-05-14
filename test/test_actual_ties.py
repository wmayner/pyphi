"""Tie-resolution behavior for actual causation.

Tests Albantakis et al. 2019 Definition 1 (actual cause maximality +
minimality) and the Exclusion principle's "indeterminate" outcome
under genuine symmetric over-determination.

The 2019 paper specifies (Definition 1, outcome 2): when multiple
minimal candidates achieve alpha_max with non-comparable purviews, the
actual cause is undetermined and the tied set is the canonical result.
This file pins that the cascade-routed implementation surfaces the
tied set as ``CausalLink.purview_ties`` (a tuple of tied
:class:`AcRepertoireIrreducibilityAnalysis` instances), in addition to
the historical ``extended_purview`` tuple-of-purviews accessor.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import actual
from pyphi import jsonify
from pyphi.models import CausalLink


def _cast_cause(result) -> CausalLink:
    """Narrow the union return type of ``find_actual_cause`` for tests.

    ``find_causal_link`` returns ``[]`` (an empty list) when no purview
    has positive alpha. Tests in this file build transitions where a
    positive-alpha actual cause is guaranteed, so we narrow to
    :class:`CausalLink` for both static checking and runtime
    correctness.
    """
    assert isinstance(result, CausalLink), (
        f"expected CausalLink, got {type(result).__name__}"
    )
    return result


@pytest.fixture
def disjunction_substrate() -> Substrate:
    """3-node substrate from 2019 Fig 7A: A, B independent inputs to OR-gate C.

    A and B have no inputs (uniform random under causal marginalization);
    C fires when A_{t-1} OR B_{t-1} is on. This is the canonical
    symmetric-over-determination fixture.
    """
    tpm = np.array(
        [
            [0.5, 0.5, 0],  # AB=00
            [0.5, 0.5, 1],  # A=1
            [0.5, 0.5, 1],  # B=1
            [0.5, 0.5, 1],  # AB=11
            [0.5, 0.5, 0],  # C=1, AB=00 (C is a downstream node)
            [0.5, 0.5, 1],
            [0.5, 0.5, 1],
            [0.5, 0.5, 1],
        ]
    )
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(tpm, cm, node_labels=("A", "B", "C"))


@pytest.fixture
def conjunction_substrate() -> Substrate:
    """3-node substrate from 2019 Fig 7B: A, B independent inputs to AND-gate D."""
    tpm = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
        ]
    )
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(tpm, cm, node_labels=("A", "B", "D"))


@pytest.fixture
def majority_substrate() -> Substrate:
    """5-node substrate from 2019 Fig 8: A,B,C,D random inputs to majority-gate M.

    M fires when sum(A_{t-1}, B_{t-1}, C_{t-1}, D_{t-1}) >= 3.
    The fully-symmetric over-determination case is state ABCD=1111→M=1.
    """
    n_nodes = 5
    n_states = 2**n_nodes
    tpm = np.zeros((n_states, n_nodes))
    for idx in range(n_states):
        a, b, c, d = ((idx >> i) & 1 for i in range(4))
        tpm[idx, 0:4] = 0.5
        tpm[idx, 4] = 1.0 if (a + b + c + d) >= 3 else 0.0
    cm = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    return Substrate(tpm, cm, node_labels=("A", "B", "C", "D", "M"))


class TestDisjunctionOverdetermination:
    """OR-gate symmetric over-determination (2019 Fig 7A).

    Inputs A=1, B=1 → C=1. Both {A=1} and {B=1} are minimal candidates
    at alpha_c = 0.415; {AB=11} also has alpha_c = 0.415 but is excluded by
    minimality (a strict superset of a tied minimal candidate). The
    actual cause is undetermined between {A=1} and {B=1}.
    """

    def _cause(self, substrate: Substrate):
        transition = actual.Transition(
            substrate,
            before_state=(1, 1, 0),
            after_state=(1, 1, 1),
            cause_indices=(0, 1),
            effect_indices=(2,),
        )
        return _cast_cause(transition.find_actual_cause((2,)))

    def test_actual_cause_alpha_is_paper_value(self, disjunction_substrate):
        cause = self._cause(disjunction_substrate)
        assert cause.alpha == pytest.approx(0.415, abs=1e-2)

    def test_extended_purview_carries_both_minimal_candidates(
        self, disjunction_substrate
    ):
        """Historical accessor: ``extended_purview`` lists tied minimal purviews."""
        cause = self._cause(disjunction_substrate)
        assert cause.extended_purview is not None
        purviews_as_sorted_tuples = {tuple(sorted(p)) for p in cause.extended_purview}
        assert purviews_as_sorted_tuples == {(0,), (1,)}

    def test_purview_ties_is_first_class_acria_set(self, disjunction_substrate):
        """Cascade-routed accessor: ``purview_ties`` exposes the tied AcRIAs.

        This is the canonical tie-set field added by the cascade
        migration. Each member is an
        :class:`AcRepertoireIrreducibilityAnalysis` carrying its own
        alpha, partition, and probabilities — strictly richer than
        ``extended_purview`` (which carries only purview tuples).
        """
        cause = self._cause(disjunction_substrate)
        ties = cause.purview_ties
        assert ties is not None
        assert len(ties) == 2
        # Each tied member is a full AcRIA.
        for tied in ties:
            assert isinstance(tied, actual.AcRepertoireIrreducibilityAnalysis)
            assert tied.alpha == pytest.approx(0.415, abs=1e-2)
        purviews = {tuple(sorted(t.purview)) for t in ties}
        assert purviews == {(0,), (1,)}

    def test_canonical_winner_is_lex_smallest_purview(self, disjunction_substrate):
        """Among tied minimal candidates, the representative purview is the
        lex-canonical (smallest under tuple ordering) one. This is a
        pyphi-specific determinism rule, not a paper rule — but it
        guarantees reproducibility under iteration-order changes."""
        cause = self._cause(disjunction_substrate)
        assert tuple(sorted(cause.purview)) == (0,)


class TestConjunctionNoTie:
    """AND-gate (2019 Fig 7B): the only minimal candidate is the joint
    occurrence {AB=11}; no symmetric over-determination."""

    def test_purview_ties_singleton(self, conjunction_substrate):
        transition = actual.Transition(
            conjunction_substrate,
            before_state=(1, 1, 0),
            after_state=(1, 1, 1),
            cause_indices=(0, 1),
            effect_indices=(2,),
        )
        cause = _cast_cause(transition.find_actual_cause((2,)))
        # The actual cause is the joint occurrence {AB=11}; no tie.
        assert tuple(sorted(cause.purview)) == (0, 1)
        # purview_ties may be None (no tie) or a 1-tuple — either is
        # acceptable, but if present must contain exactly the winner.
        if cause.purview_ties is not None:
            assert len(cause.purview_ties) == 1
            assert cause.purview_ties[0].purview == cause.purview


class TestMajoritySymmetricOverdetermination:
    """Majority gate fully-symmetric case (2019 p. 19): ABCD=1111→M=1
    has four tied third-order minimal candidates {ABC}, {ABD},
    {ACD}, {BCD}."""

    def test_purview_ties_has_four_minimal_third_order_candidates(
        self, majority_substrate
    ):
        transition = actual.Transition(
            majority_substrate,
            before_state=(1, 1, 1, 1, 0),
            after_state=(1, 1, 1, 1, 1),
            cause_indices=(0, 1, 2, 3),
            effect_indices=(4,),
        )
        cause = _cast_cause(transition.find_actual_cause((4,)))
        ties = cause.purview_ties
        assert ties is not None
        # Four three-input subsets of {A,B,C,D}.
        purviews = {tuple(sorted(t.purview)) for t in ties}
        expected = {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
        assert purviews == expected
        # All tied members have identical alpha.
        alphas = {round(t.alpha, 6) for t in ties}
        assert len(alphas) == 1


class TestACDeterminism:
    """50 repeated invocations of the OR-gate transition produce the
    same representative purview and the same tied-set membership.
    Guards against iteration-order drift in ``potential_purviews`` and
    against future regressions in the cascade's determinism fallback.
    """

    def test_disjunction_50_iterations_stable(self, disjunction_substrate):
        results = []
        for _ in range(50):
            transition = actual.Transition(
                disjunction_substrate,
                before_state=(1, 1, 0),
                after_state=(1, 1, 1),
                cause_indices=(0, 1),
                effect_indices=(2,),
            )
            results.append(_cast_cause(transition.find_actual_cause((2,))))
        baseline = results[0]
        baseline_purview = tuple(sorted(baseline.purview))
        baseline_tied_purviews = (
            tuple(sorted(tuple(sorted(t.purview)) for t in baseline.purview_ties))
            if baseline.purview_ties is not None
            else None
        )
        for idx, r in enumerate(results[1:], start=1):
            assert tuple(sorted(r.purview)) == baseline_purview, (
                f"iteration {idx} chose different representative purview"
            )
            tied = (
                tuple(sorted(tuple(sorted(t.purview)) for t in r.purview_ties))
                if r.purview_ties is not None
                else None
            )
            assert tied == baseline_tied_purviews, (
                f"iteration {idx} produced different tied set"
            )


class TestTieRoundTrip:
    """JSON round-trip preserves the tied AcRIA set on
    :class:`CausalLink`. Mirrors the C.7 round-trip pattern for RIA."""

    def test_purview_ties_survive_round_trip(self, disjunction_substrate):
        transition = actual.Transition(
            disjunction_substrate,
            before_state=(1, 1, 0),
            after_state=(1, 1, 1),
            cause_indices=(0, 1),
            effect_indices=(2,),
        )
        cause = _cast_cause(transition.find_actual_cause((2,)))
        assert cause.purview_ties is not None
        encoded = jsonify.dumps(cause)
        decoded = jsonify.loads(encoded)
        assert isinstance(decoded, CausalLink)
        assert decoded.purview_ties is not None
        assert len(decoded.purview_ties) == len(cause.purview_ties)
        decoded_purviews = {tuple(sorted(t.purview)) for t in decoded.purview_ties}
        original_purviews = {tuple(sorted(t.purview)) for t in cause.purview_ties}
        assert decoded_purviews == original_purviews
