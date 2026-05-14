"""Property-based invariants for the cascade tie-resolution primitive.

Covers structural invariants of ``cascade``, ``resolve_state_tie``, and
``resolve_complex_tie`` over Hypothesis-generated synthetic inputs, plus
a parallel-determinism stress test on the AND-XOR substrate (where the
SIA state-tie cascade is exercised).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import config
from pyphi import resolve_ties
from pyphi.resolve_ties import CascadeLevel
from pyphi.resolve_ties import NotAComplex
from pyphi.resolve_ties import ResolutionContext
from pyphi.resolve_ties import cascade
from pyphi.system import System

from . import example_substrates


@dataclass(frozen=True)
class FakeCandidate:
    """Minimal cascade candidate with two cascade keys and a label."""

    label: str
    phi: float = 0.0
    big_phi: float = 0.0


_candidate_strategy = st.builds(
    FakeCandidate,
    label=st.text(min_size=1, max_size=4),
    phi=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    big_phi=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)

_candidate_list_strategy = st.lists(_candidate_strategy, min_size=1, max_size=8)


def _phi_level() -> CascadeLevel:
    return CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi)


def _big_phi_level() -> CascadeLevel:
    return CascadeLevel(postulate="Composition", op="argmax", key=lambda c: c.big_phi)


_HEALTH_CHECKS = [HealthCheck.too_slow, HealthCheck.data_too_large]


class TestCascadeStructuralInvariants:
    """Structural invariants that hold for any non-empty candidate list."""

    @settings(max_examples=80, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(candidates=_candidate_list_strategy)
    def test_resolved_winner_is_in_input_set(self, candidates):
        ctx = ResolutionContext(max_escalation_level="Composition")
        outcome = cascade(
            candidates,
            levels=[_phi_level(), _big_phi_level()],
            context=ctx,
            on_unresolved="defer",
        )
        if outcome.resolved is not None:
            assert outcome.resolved in candidates

    @settings(max_examples=80, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(candidates=_candidate_list_strategy)
    def test_tied_set_is_subset_of_input(self, candidates):
        ctx = ResolutionContext(max_escalation_level="Composition")
        outcome = cascade(
            candidates,
            levels=[_phi_level(), _big_phi_level()],
            context=ctx,
            on_unresolved="defer",
        )
        for member in outcome.tied_set:
            assert member in candidates

    @settings(max_examples=80, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(candidates=_candidate_list_strategy)
    def test_resolved_winner_attains_max_phi(self, candidates):
        ctx = ResolutionContext(max_escalation_level="Integration")
        outcome = cascade(
            candidates,
            levels=[_phi_level()],
            context=ctx,
            on_unresolved="defer",
        )
        max_phi = max(c.phi for c in candidates)
        if outcome.resolved is not None:
            assert outcome.resolved.phi == max_phi
        if outcome.outcome == "UNRESOLVED_WITHIN_BUDGET":
            # When unresolved, every survivor must attain max phi at the
            # final processed level.
            for member in outcome.tied_set:
                assert member.phi == max_phi

    @settings(max_examples=80, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(
        candidates=_candidate_list_strategy,
        perm_seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_resolved_winner_unique_under_permutation(self, candidates, perm_seed):
        """Input order does not change the resolved winner *or* the
        membership of the tied set."""
        import random as stdlib_random

        rng = stdlib_random.Random(perm_seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        ctx_a = ResolutionContext(max_escalation_level="Composition")
        ctx_b = ResolutionContext(max_escalation_level="Composition")
        outcome_a = cascade(
            candidates, levels=[_phi_level(), _big_phi_level()], context=ctx_a
        )
        outcome_b = cascade(
            shuffled, levels=[_phi_level(), _big_phi_level()], context=ctx_b
        )
        # When a unique winner exists, identity must match across orderings.
        if outcome_a.outcome == "RESOLVED" and outcome_b.outcome == "RESOLVED":
            assert outcome_a.resolved == outcome_b.resolved
        # The tied set (as a multiset of members) must always match.
        assert sorted(outcome_a.tied_set, key=id) == sorted(outcome_b.tied_set, key=id)


class TestCascadeBudgetEscalation:
    """Escalation budget controls how far the cascade walks."""

    @settings(max_examples=40, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(n=st.integers(min_value=2, max_value=6))
    def test_budget_blocks_escalation_past_integration(self, n):
        """With ``max_escalation_level='Integration'``, the cascade does
        not consult Composition-level keys even when Integration ties."""
        # All candidates tied at phi=1.0 but differ at big_phi.
        candidates = [
            FakeCandidate(label=str(i), phi=1.0, big_phi=float(i)) for i in range(n)
        ]
        ctx = ResolutionContext(max_escalation_level="Integration")
        outcome = cascade(
            candidates,
            levels=[_phi_level(), _big_phi_level()],
            context=ctx,
            on_unresolved="defer",
        )
        assert outcome.outcome == "UNRESOLVED_WITHIN_BUDGET"
        assert len(outcome.tied_set) == n

    @settings(max_examples=40, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(n=st.integers(min_value=2, max_value=6))
    def test_full_budget_escalates_to_composition(self, n):
        """With ``max_escalation_level='Composition'``, the same input
        resolves at Composition when big_phi values are distinct."""
        candidates = [
            FakeCandidate(label=str(i), phi=1.0, big_phi=float(i)) for i in range(n)
        ]
        ctx = ResolutionContext(max_escalation_level="Composition")
        outcome = cascade(
            candidates,
            levels=[_phi_level(), _big_phi_level()],
            context=ctx,
        )
        assert outcome.outcome == "RESOLVED"
        assert outcome.resolved is not None
        assert outcome.resolved.big_phi == float(n - 1)
        assert outcome.cascade_level == "Composition"

    def test_full_tie_with_on_unresolved_fail_raises(self):
        """All keys tied: ``on_unresolved='fail'`` raises NotAComplex."""
        candidates = [
            FakeCandidate(label="a", phi=1.0, big_phi=2.0),
            FakeCandidate(label="b", phi=1.0, big_phi=2.0),
        ]
        ctx = ResolutionContext(max_escalation_level="Composition")
        with pytest.raises(NotAComplex):
            cascade(
                candidates,
                levels=[_phi_level(), _big_phi_level()],
                context=ctx,
                on_unresolved="fail",
            )


class TestCascadeMonotonicity:
    """If the cascade resolves at level K, all earlier levels must have
    been tied across the candidate set up to K."""

    @settings(max_examples=60, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(candidates=st.lists(_candidate_strategy, min_size=2, max_size=8))
    def test_resolution_at_composition_implies_phi_tied(self, candidates):
        """When the cascade reports ``cascade_level='Composition'``, all
        survivors up to that level were tied at the Integration key."""
        ctx = ResolutionContext(max_escalation_level="Composition")
        outcome = cascade(
            candidates,
            levels=[_phi_level(), _big_phi_level()],
            context=ctx,
            on_unresolved="defer",
        )
        if outcome.outcome == "RESOLVED" and outcome.cascade_level == "Composition":
            phi_values = {c.phi for c in outcome.tied_set}
            assert len(phi_values) == 1, (
                "Composition-level resolution requires all tied-set members "
                f"to share a phi value; got {phi_values}"
            )


class TestSubstrateExclusionCascade:
    """``resolve_complex_tie`` walks the Composition step on
    pre-filtered overlap cliques."""

    @settings(max_examples=40, deadline=None, suppress_health_check=_HEALTH_CHECKS)
    @given(n=st.integers(min_value=2, max_value=5))
    def test_unique_big_phi_winner_resolves(self, n):
        candidates = [
            FakeCandidate(label=str(i), phi=1.0, big_phi=float(i)) for i in range(n)
        ]
        ctx = ResolutionContext(max_escalation_level="Composition")
        outcome = resolve_ties.resolve_complex_tie(candidates, context=ctx)
        assert outcome.outcome == "RESOLVED"
        assert outcome.resolved is not None
        assert outcome.resolved.big_phi == float(n - 1)


def _and_xor_system() -> System:
    return System(example_substrates.and_xor_substrate(), (0, 1))


def _run_sia_n_times(system: System, n: int, parallel: bool) -> list:
    with config.override(parallel=parallel):
        return [system.sia() for _ in range(n)]


class TestStateTieStressDeterminism:
    """Repeated invocations of ``sia()`` on a substrate with state ties
    produce identical results. Exercises the state-tie cascade hot path
    under both sequential and parallel execution."""

    @pytest.mark.slow
    def test_and_xor_sequential_stress(self):
        results = _run_sia_n_times(_and_xor_system(), n=50, parallel=False)
        baseline = results[0]
        for idx, r in enumerate(results[1:], start=1):
            assert r == baseline, (
                f"Iteration {idx} diverged from baseline:\n"
                f"  baseline phi={baseline.phi}\n"
                f"  iteration phi={r.phi}"
            )

    @pytest.mark.slow
    def test_and_xor_parallel_stress(self):
        results = _run_sia_n_times(_and_xor_system(), n=50, parallel=True)
        baseline = results[0]
        for idx, r in enumerate(results[1:], start=1):
            assert r == baseline, (
                f"Iteration {idx} (parallel) diverged from baseline:\n"
                f"  baseline phi={baseline.phi}\n"
                f"  iteration phi={r.phi}"
            )

    def test_and_xor_sequential_equals_parallel(self):
        seq = _run_sia_n_times(_and_xor_system(), n=1, parallel=False)[0]
        par = _run_sia_n_times(_and_xor_system(), n=1, parallel=True)[0]
        assert seq == par
        assert seq.phi == par.phi
