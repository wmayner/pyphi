"""Unit tests for the exclusion cascade over Candidate records."""

import pytest

from pyphi.condensation import Candidate
from pyphi.condensation import PendingCandidate
from pyphi.condensation import exclusion_cascade
from pyphi.condensation import exclusion_records
from pyphi.condensation import gated_exclusion_cascade


class _StubSystem:
    """System stand-in: fingerprint + counted ces() with a fixed big Φ."""

    calls = 0

    def __init__(self, big_phi, fingerprint):
        self._big_phi = big_phi
        self._fingerprint = fingerprint

    def ces(self):
        type(self).calls += 1
        outer = self

        class _CES:
            big_phi = outer._big_phi

        return _CES()


def _candidate(footprint, phi, big_phi=0.0, fingerprint=None):
    system = _StubSystem(big_phi, fingerprint if fingerprint is not None else object())
    return Candidate(
        footprint=frozenset(footprint),
        phi=phi,
        sia_provider=lambda: None,
        system_provider=lambda: system,
    )


def _footprints(outcome):
    return [tuple(sorted(c.footprint)) for c in outcome.accepted]


def test_chain_recursion_accepts_disjoint_lower_candidate():
    """A candidate beaten only by *excluded* rivals is a complex.

    Chain: {0,1} phi=3 overlaps {1,2} phi=2 overlaps {2,3} phi=1;
    {0,1} and {2,3} are disjoint. Recursive carving yields both.
    """
    candidates = [
        _candidate({0, 1}, 3.0),
        _candidate({1, 2}, 2.0),
        _candidate({2, 3}, 1.0),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1), (2, 3)]
    assert outcome.failed_cliques == ()


def test_tied_clique_escalates_to_big_phi():
    """phi-tied overlapping candidates resolve by big Φ."""
    winner = _candidate({0, 1}, 1.0, big_phi=5.0)
    loser = _candidate({1, 2}, 1.0, big_phi=3.0)
    outcome = exclusion_cascade([winner, loser])
    assert _footprints(outcome) == [(0, 1)]


def test_phi_tied_clique_fails_exclusion_and_units_stay_available():
    """A Φ-tied clique is removed; its units remain available below."""
    a = _candidate({0, 1}, 1.0, big_phi=2.0)
    b = _candidate({1, 2}, 1.0, big_phi=2.0)
    lower = _candidate({0}, 0.5)
    outcome = exclusion_cascade([a, b, lower])
    assert _footprints(outcome) == [(0,)]
    assert len(outcome.failed_cliques) == 1
    assert {tuple(sorted(c.footprint)) for c in outcome.failed_cliques[0]} == {
        (0, 1),
        (1, 2),
    }


def test_single_fingerprint_clique_skips_escalation():
    """Identical kernel fingerprints ⇒ Φ ties by bit-identity: no ces() runs."""
    _StubSystem.calls = 0
    fp = b"same-digest"
    a = _candidate({0, 1}, 1.0, big_phi=2.0, fingerprint=fp)
    b = _candidate({1, 2}, 1.0, big_phi=2.0, fingerprint=fp)
    outcome = exclusion_cascade([a, b])
    assert outcome.accepted == ()
    assert len(outcome.failed_cliques) == 1
    assert _StubSystem.calls == 0


def test_mixed_fingerprint_clique_computes_big_phi_once_per_fingerprint():
    _StubSystem.calls = 0
    fp = b"shared"
    a = _candidate({0, 1}, 1.0, big_phi=2.0, fingerprint=fp)
    b = _candidate({1, 2}, 1.0, big_phi=2.0, fingerprint=fp)
    c = _candidate({0, 2}, 1.0, big_phi=7.0, fingerprint=b"other")
    outcome = exclusion_cascade([a, b, c])
    assert _footprints(outcome) == [(0, 2)]
    assert _StubSystem.calls == 2  # one per distinct fingerprint, not per member


def test_exclusion_records_key_on_footprints():
    top = _candidate({0, 1}, 3.0)
    beaten = _candidate({1, 2}, 2.0)
    disjoint = _candidate({2, 3}, 1.0)
    candidates = [top, beaten, disjoint]
    outcome = exclusion_cascade(candidates)
    records = exclusion_records(outcome.accepted, candidates)
    assert {r.node_indices for r in records[(0, 1)]} == {(1, 2)}
    assert {r.node_indices for r in records[(2, 3)]} == {(1, 2)}


def test_noise_level_big_phi_difference_ties_at_precision():
    """Φ differing only in the last ulps is a tie, not a resolution."""
    a = _candidate({0, 1}, 1.0, big_phi=0.9726647808729815, fingerprint=b"x")
    b = _candidate({1, 2}, 1.0, big_phi=0.9726647808729809, fingerprint=b"y")
    outcome = exclusion_cascade([a, b])
    assert outcome.accepted == ()
    assert len(outcome.failed_cliques) == 1


def test_exclusion_records_include_same_footprint_rivals():
    """A losing candidate on the winner's exact footprint is a genuinely
    excluded rival (macro door: a rival grain over the same micro units)
    and must appear in the winner's records."""
    winner = _candidate({0, 1}, 2.0)
    rival_grain = _candidate({0, 1}, 1.0)
    candidates = [winner, rival_grain]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1)]
    records = exclusion_records(outcome.accepted, candidates)
    assert [(r.node_indices, r.phi) for r in records[(0, 1)]] == [((0, 1), 1.0)]


def test_tied_chain_accepts_disjoint_end_beaten_only_by_excluded_rival():
    """φ-tied chain: the Φ-max end wins its conflicts; the far end overlaps
    only the excluded middle and must be accepted (Marshall et al. 2023,
    Alg. A1 tied branch)."""
    candidates = [
        _candidate({0, 1}, 1.0, big_phi=5.0),
        _candidate({1, 2}, 1.0, big_phi=4.0),
        _candidate({2, 3}, 1.0, big_phi=3.0),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1), (2, 3)]
    assert outcome.failed_cliques == ()


def test_tied_chain_middle_winner_excludes_both_ends():
    """φ-tied chain where the middle is Φ-max: both ends overlap the winner."""
    candidates = [
        _candidate({0, 1}, 1.0, big_phi=2.0),
        _candidate({1, 2}, 1.0, big_phi=9.0),
        _candidate({2, 3}, 1.0, big_phi=2.0),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(1, 2)]
    assert outcome.failed_cliques == ()


def test_tied_chain_disjoint_phi_max_pair_both_accepted():
    """A Φ tie between candidates that do NOT overlap each other is not an
    exclusion conflict: both are complexes; only the overlapping loser falls."""
    candidates = [
        _candidate({0, 1}, 1.0, big_phi=9.0),
        _candidate({1, 2}, 1.0, big_phi=2.0),
        _candidate({2, 3}, 1.0, big_phi=9.0),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1), (2, 3)]
    assert outcome.failed_cliques == ()


def test_tied_isolated_candidates_never_escalate_to_big_phi():
    """φ-tied candidates with no overlap conflict are accepted without any
    cause-effect-structure computation."""

    class _ExplodingSystem:
        _fingerprint = b"exploding"

        def ces(self):
            raise AssertionError("Φ escalation must not run for isolated candidates")

    exploding = _ExplodingSystem()
    candidates = [
        Candidate(
            footprint=frozenset({0, 1}),
            phi=1.0,
            sia_provider=lambda: None,
            system_provider=lambda: exploding,
        ),
        Candidate(
            footprint=frozenset({2, 3}),
            phi=1.0,
            sia_provider=lambda: None,
            system_provider=lambda: exploding,
        ),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1), (2, 3)]
    assert outcome.failed_cliques == ()


def _lazy(candidates, ceilings):
    """(pending, evaluate_batch, calls): each pending resolves to its
    candidate; calls records each forced band's footprints."""
    by_id = {}
    pending = []
    for cand, ceiling in zip(candidates, ceilings, strict=True):
        p = PendingCandidate(footprint=cand.footprint, ceiling=ceiling, payload=cand)
        by_id[id(p)] = cand
        pending.append(p)
    calls = []

    def evaluate_batch(band):
        calls.append([tuple(sorted(p.footprint)) for p in band])
        return [by_id[id(p)] for p in band]

    return pending, evaluate_batch, calls


class TestGatedExclusionCascade:
    def test_matches_eager_cascade_when_everything_forced(self):
        # Loose ceilings put every candidate in the first forced band, so
        # nothing is gated and the outcome must match the eager cascade
        # exactly (accepted: {0,1} then {3}; {1,2} and {2,3} drop by
        # coverage).
        candidates = [
            _candidate({0, 1}, 1.0),
            _candidate({1, 2}, 0.5),
            _candidate({3}, 0.25),
            _candidate({2, 3}, 0.1),
        ]
        eager = exclusion_cascade(candidates)
        pending, evaluate_batch, _ = _lazy(candidates, [1.0] * 4)
        outcome, gated = gated_exclusion_cascade(pending, evaluate_batch)
        assert outcome.accepted == eager.accepted
        assert outcome.failed_cliques == eager.failed_cliques
        assert [tuple(sorted(c.footprint)) for c in outcome.accepted] == [
            (0, 1),
            (3,),
        ]
        assert gated == ()

    def test_tight_ceilings_gate_coverage_dropped_candidates(self):
        # With ceilings equal to each phi, candidates the eager cascade
        # drops by coverage are certified skips instead: same accepted
        # set, but {1,2} and {2,3} are never evaluated.
        candidates = [
            _candidate({0, 1}, 1.0),
            _candidate({1, 2}, 0.5),
            _candidate({3}, 0.25),
            _candidate({2, 3}, 0.1),
        ]
        eager = exclusion_cascade(candidates)
        pending, evaluate_batch, calls = _lazy(candidates, [c.phi for c in candidates])
        outcome, gated = gated_exclusion_cascade(pending, evaluate_batch)
        assert outcome.accepted == eager.accepted
        assert outcome.failed_cliques == eager.failed_cliques
        assert sorted(tuple(sorted(p.footprint)) for p in gated) == [
            (1, 2),
            (2, 3),
        ]
        forced = [fp for band in calls for fp in band]
        assert (1, 2) not in forced
        assert (2, 3) not in forced

    def test_gates_overlapping_candidate_below_ceiling(self):
        candidates = [
            _candidate({0, 1}, 1.0),
            _candidate({1, 2}, 0.5),  # overlaps winner, ceiling below
            _candidate({3}, 0.25),  # disjoint: must still be evaluated
        ]
        pending, evaluate_batch, calls = _lazy(candidates, [1.0, 0.6, 0.3])
        outcome, gated = gated_exclusion_cascade(pending, evaluate_batch)
        assert [tuple(sorted(c.footprint)) for c in outcome.accepted] == [
            (0, 1),
            (3,),
        ]
        assert [tuple(sorted(p.footprint)) for p in gated] == [(1, 2)]
        forced = [fp for band in calls for fp in band]
        assert (1, 2) not in forced
        assert (3,) in forced

    def test_disjoint_low_candidate_never_gated_by_global_max(self):
        candidates = [
            _candidate({0, 1}, 1.0),
            _candidate({2, 3}, 0.01),
        ]
        pending, evaluate_batch, _ = _lazy(candidates, [1.0, 0.02])
        outcome, gated = gated_exclusion_cascade(pending, evaluate_batch)
        assert len(outcome.accepted) == 2
        assert gated == ()

    def test_tolerance_tied_pending_is_forced_not_gated(self):
        # Overlapping candidates whose ceilings tie the winner at precision
        # must be evaluated so the tie machinery sees them.
        shared = object()
        candidates = [
            _candidate({0, 1}, 1.0, fingerprint=shared),
            _candidate({1, 2}, 1.0, fingerprint=shared),
        ]
        eager = exclusion_cascade(candidates)
        pending, evaluate_batch, _ = _lazy(candidates, [1.0, 1.0])
        outcome, gated = gated_exclusion_cascade(pending, evaluate_batch)
        assert gated == ()
        # Same-fingerprint overlapping tie: both fail exclusion.
        assert outcome.accepted == eager.accepted == ()
        assert len(outcome.failed_cliques) == len(eager.failed_cliques) == 1

    def test_ceiling_violation_raises(self):
        candidates = [_candidate({0, 1}, 1.0)]
        pending, evaluate_batch, _ = _lazy(candidates, [0.5])  # phi > ceiling
        with pytest.raises(RuntimeError, match="ceiling"):
            gated_exclusion_cascade(pending, evaluate_batch)

    def test_empty_pending(self):
        outcome, gated = gated_exclusion_cascade([], lambda _band: [])
        assert outcome.accepted == ()
        assert outcome.failed_cliques == ()
        assert gated == ()
