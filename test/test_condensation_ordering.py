"""The exclusion cascade's φₛ-tier clustering and within-tier ordering."""

from types import SimpleNamespace

from pyphi.condensation import Candidate
from pyphi.condensation import exclusion_cascade

NOISE = 5.6e-16


def _candidate(footprint, phi, tag):
    return Candidate(
        footprint=frozenset(footprint),
        phi=phi,
        sia_provider=lambda tag=tag: tag,
        system_provider=lambda tag=tag: tag,
    )


def _stub_candidate(footprint, phi, big_phi):
    """A candidate whose ``system_provider`` yields a system with a real
    ``ces().big_phi`` but no ``_fingerprint`` (so Φ is computed, not
    deduplicated)."""
    system = SimpleNamespace(
        ces=lambda big_phi=big_phi: SimpleNamespace(big_phi=big_phi)
    )
    return Candidate(
        footprint=frozenset(footprint),
        phi=phi,
        sia_provider=lambda system=system: system,
        system_provider=lambda system=system: system,
    )


class TestTierOrdering:
    def test_disjoint_noise_tied_candidates_keep_input_order(self):
        # Two disjoint candidates tied up to noise: BOTH are accepted and
        # their order (hence which is "maximal") follows input order, not
        # the bit pattern of the noise.
        a = _candidate({0, 1}, 0.3, "a")
        b = _candidate({2, 3}, 0.3 + NOISE, "b")  # bitwise larger

        accepted_ab = exclusion_cascade([a, b]).accepted
        accepted_ba = exclusion_cascade([b, a]).accepted

        assert [c.sia_provider() for c in accepted_ab] == ["a", "b"]
        assert [c.sia_provider() for c in accepted_ba] == ["b", "a"]

    def test_unsorted_input_is_tiered_correctly(self):
        # Callers no longer pre-sort; a genuinely lower-phi candidate in
        # front must still land in the later tier.
        low = _candidate({0, 1}, 0.1, "low")
        high = _candidate({0, 1}, 0.9, "high")  # overlaps low
        outcome = exclusion_cascade([low, high])
        assert [c.sia_provider() for c in outcome.accepted] == ["high"]

    def test_tier_membership_is_tolerant(self):
        # Overlapping candidates tied up to noise form ONE clique (tier
        # co-membership), so the Composition escalation runs. With Φ also
        # tied to noise the clique fails exclusion and NEITHER is accepted;
        # were the noise to split the tier, the leading candidate would be
        # accepted alone. Asserting zero accepted therefore proves tolerant
        # co-membership.
        a = _stub_candidate({0, 1}, 0.3, 0.5)
        b = _stub_candidate({1, 2}, 0.3 + NOISE, 0.5 + NOISE)  # overlaps a
        outcome = exclusion_cascade([a, b])
        assert len(outcome.accepted) == 0
