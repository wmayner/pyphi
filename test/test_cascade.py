"""Tests for the cascade tie-resolution primitive.

The cascade walks the IIT postulate hierarchy (Existence → ... →
Composition) to resolve ties at the lowest sufficient postulate. The
``ResolutionContext`` carries the entry-point's escalation budget and
memoization caches.

See ``docs/superpowers/specs/2026-05-13-cascade-execution-model.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyphi.resolve_ties import CascadeLevel
from pyphi.resolve_ties import NotAComplex
from pyphi.resolve_ties import ResolutionContext
from pyphi.resolve_ties import cascade

# ----- Postulate ordering -----


def test_can_escalate_to_returns_true_for_lower_or_equal_postulate():
    """ResolutionContext respects postulate order via can_escalate_to."""
    ctx = ResolutionContext(max_escalation_level="Integration")
    assert ctx.can_escalate_to("Information")
    assert ctx.can_escalate_to("Integration")


def test_can_escalate_to_returns_false_for_higher_postulate():
    """ResolutionContext refuses escalation past its budget."""
    ctx = ResolutionContext(max_escalation_level="Integration")
    assert not ctx.can_escalate_to("Composition")
    assert not ctx.can_escalate_to("Exclusion")


def test_can_escalate_to_full_budget_allows_all():
    """Composition-budget context permits any postulate."""
    ctx = ResolutionContext(max_escalation_level="Composition")
    assert ctx.can_escalate_to("Information")
    assert ctx.can_escalate_to("Integration")
    assert ctx.can_escalate_to("Exclusion")
    assert ctx.can_escalate_to("Composition")


# ----- Memoization -----


def test_memoize_computes_once_per_key():
    """ResolutionContext.memoize caches by key; second call doesn't re-run fn."""
    ctx = ResolutionContext(max_escalation_level="Composition")
    call_count = [0]

    def expensive():
        call_count[0] += 1
        return 42

    assert ctx.memoize("k1", expensive) == 42
    assert ctx.memoize("k1", expensive) == 42
    assert call_count[0] == 1


def test_memoize_distinct_keys_compute_separately():
    """Different memo keys cache independently."""
    ctx = ResolutionContext(max_escalation_level="Composition")
    call_count = [0]

    def increment_and_return():
        call_count[0] += 1
        return call_count[0]

    a = ctx.memoize("a", increment_and_return)
    b = ctx.memoize("b", increment_and_return)
    assert (a, b) == (1, 2)


def test_child_context_inherits_budget_and_memo():
    """A child context reuses the parent's escalation budget and memo cache."""
    parent = ResolutionContext(max_escalation_level="Composition")
    parent.memoize("shared", lambda: "value")
    child = parent.child()
    assert child.max_escalation_level == "Composition"
    # Should hit the parent's cache, not re-run.
    sentinel_calls = [0]

    def should_not_run():
        sentinel_calls[0] += 1
        return "WRONG"

    assert child.memoize("shared", should_not_run) == "value"
    assert sentinel_calls[0] == 0


# ----- Cascade walks -----


@dataclass(frozen=True)
class FakePhiObject:
    """Minimal stand-in for RIA/SIA/MICE in cascade unit tests."""

    label: str
    phi: float = 0.0
    big_phi: float = 0.0


def test_cascade_resolves_at_first_level_with_unique_winner():
    """Single argmax level with a unique winner returns RESOLVED."""
    candidates = [FakePhiObject("a", phi=1.0), FakePhiObject("b", phi=2.0)]
    ctx = ResolutionContext(max_escalation_level="Integration")
    outcome = cascade(
        candidates,
        levels=[CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi)],
        context=ctx,
    )
    assert outcome.outcome == "RESOLVED"
    assert outcome.resolved is candidates[1]
    assert outcome.cascade_level == "Integration"


def test_cascade_walks_to_second_level_when_first_ties():
    """Tie at level 1 escalates to level 2; unique winner there resolves."""
    candidates = [
        FakePhiObject("a", phi=2.0, big_phi=1.0),
        FakePhiObject("b", phi=2.0, big_phi=3.0),
        FakePhiObject("c", phi=1.0, big_phi=10.0),  # not tied at phi
    ]
    ctx = ResolutionContext(max_escalation_level="Composition")
    outcome = cascade(
        candidates,
        levels=[
            CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi),
            CascadeLevel(postulate="Composition", op="argmax", key=lambda c: c.big_phi),
        ],
        context=ctx,
    )
    assert outcome.outcome == "RESOLVED"
    assert outcome.resolved is candidates[1]
    assert outcome.cascade_level == "Composition"


def test_cascade_stops_at_budget_when_higher_levels_would_escalate():
    """If a tie remains at the budget boundary, return UNRESOLVED_WITHIN_BUDGET."""
    candidates = [
        FakePhiObject("a", phi=2.0, big_phi=1.0),
        FakePhiObject("b", phi=2.0, big_phi=3.0),
    ]
    ctx = ResolutionContext(max_escalation_level="Integration")
    outcome = cascade(
        candidates,
        levels=[
            CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi),
            CascadeLevel(postulate="Composition", op="argmax", key=lambda c: c.big_phi),
        ],
        context=ctx,
    )
    assert outcome.outcome == "UNRESOLVED_WITHIN_BUDGET"
    assert outcome.resolved is None
    assert outcome.tied_set == tuple(candidates)
    assert outcome.cascade_level == "Integration"


def test_cascade_fails_when_all_levels_tie_and_on_unresolved_is_fail():
    """on_unresolved='fail' raises NotAComplex if cascade exhausts levels with a tie."""
    candidates = [
        FakePhiObject("a", phi=2.0, big_phi=3.0),
        FakePhiObject("b", phi=2.0, big_phi=3.0),
    ]
    ctx = ResolutionContext(max_escalation_level="Composition")
    with pytest.raises(NotAComplex) as exc_info:
        cascade(
            candidates,
            levels=[
                CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi),
                CascadeLevel(
                    postulate="Composition", op="argmax", key=lambda c: c.big_phi
                ),
            ],
            context=ctx,
            on_unresolved="fail",
        )
    assert tuple(exc_info.value.tied_set) == tuple(candidates)
    assert exc_info.value.cascade_level == "Composition"


def test_cascade_argmin_op_picks_smallest_key_value():
    """argmin works symmetrically to argmax."""
    candidates = [FakePhiObject("a", phi=3.0), FakePhiObject("b", phi=1.0)]
    ctx = ResolutionContext(max_escalation_level="Integration")
    outcome = cascade(
        candidates,
        levels=[CascadeLevel(postulate="Integration", op="argmin", key=lambda c: c.phi)],
        context=ctx,
    )
    assert outcome.resolved is candidates[1]


def test_cascade_with_single_candidate_returns_resolved_without_walking_levels():
    """Trivial case: one candidate is already resolved at any level."""
    candidates = [FakePhiObject("only", phi=1.0)]
    ctx = ResolutionContext(max_escalation_level="Composition")
    outcome = cascade(
        candidates,
        levels=[CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi)],
        context=ctx,
    )
    assert outcome.outcome == "RESOLVED"
    assert outcome.resolved is candidates[0]


def test_cascade_empty_candidates_raises():
    """Passing an empty candidate set is a programming error."""
    ctx = ResolutionContext(max_escalation_level="Integration")
    with pytest.raises(ValueError):
        cascade(
            [],
            levels=[CascadeLevel(postulate="Integration", op="argmax", key=lambda _: 0)],
            context=ctx,
        )


def test_cascade_outcome_includes_tied_set_at_each_level():
    """The tied_set on the outcome reflects candidates surviving at the resolved level."""
    candidates = [
        FakePhiObject("a", phi=2.0, big_phi=1.0),
        FakePhiObject("b", phi=2.0, big_phi=3.0),
        FakePhiObject("c", phi=1.0, big_phi=10.0),
    ]
    ctx = ResolutionContext(max_escalation_level="Composition")
    outcome = cascade(
        candidates,
        levels=[
            CascadeLevel(postulate="Integration", op="argmax", key=lambda c: c.phi),
            CascadeLevel(postulate="Composition", op="argmax", key=lambda c: c.big_phi),
        ],
        context=ctx,
    )
    # tied_set is the set entering the resolving level: (a, b) at Composition.
    assert outcome.tied_set == (candidates[0], candidates[1])


# ----- Cascade interaction with memoization -----


def test_cascade_calls_key_function_once_per_candidate_per_level():
    """The cascade does not re-call a key function for already-evaluated candidates."""
    candidates = [FakePhiObject("a", phi=1.0), FakePhiObject("b", phi=2.0)]
    call_counts: dict[str, int] = {"phi": 0}

    def phi_key(c):
        call_counts["phi"] += 1
        return c.phi

    ctx = ResolutionContext(max_escalation_level="Composition")
    cascade(
        candidates,
        levels=[CascadeLevel(postulate="Integration", op="argmax", key=phi_key)],
        context=ctx,
    )
    assert call_counts["phi"] == 2
