"""Unit tests for the exclusion cascade over Candidate records."""

from pyphi.condensation import Candidate
from pyphi.condensation import exclusion_cascade
from pyphi.condensation import exclusion_records


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
