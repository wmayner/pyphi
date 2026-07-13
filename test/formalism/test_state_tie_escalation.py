"""State-tie resolution: the S1 ladder with the IIT 4.0 (2026) cap.

Per Albantakis et al. (2023) S1: specified-state ties resolve by the
subsequent postulates in order — Integration (φ_s, the capped value under
2026), then Composition (Φ). A Φ tie among readings whose structures are
intrinsically identical (isomorphic up to unit relabeling) is extrinsic and
not a violation; a Φ tie among genuinely distinct structures violates the
information postulate and the system does not qualify as a complex. Readings
that all fail integration (φ_s = 0) leave nothing for Φ to adjudicate: the
choice among them is presentational and never triggers a cause-effect
structure computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.models.explanation import NullResultReason
from pyphi.utils import all_states


def _noisy_xor_substrate(p: float = 0.85) -> Substrate:
    """Three-unit noisy XOR loop: each unit computes the XOR of the other two
    correctly with probability ``p``. At (0, 0, 0) the specified cause states
    (0, 0, 0) and (1, 1, 1) tie by symmetry, and the capped φ_s is positive
    (the effect-side i_diff binds at the same value for both readings)."""
    n = 3
    tpm = np.zeros((2**n, n))
    for i, s in enumerate(all_states((2,) * n)):
        for j in range(n):
            xor = (sum(s) - s[j]) % 2
            tpm[i, j] = p if xor == 1 else 1 - p
    return Substrate(tpm, cm=np.ones((n, n), dtype=int) - np.eye(n, dtype=int))


@pytest.fixture
def noisy_xor_system() -> System:
    return System(_noisy_xor_substrate(), (0, 0, 0))


def test_zero_capped_state_tie_skips_composition_escalation(monkeypatch):
    """When every tied reading caps to φ_s = 0, the system is not a complex
    under any of them; the choice is presentational and must not pay for a
    cause-effect structure."""
    from pyphi import examples
    from pyphi.formalism import iit3

    def explode(*args, **kwargs):
        raise AssertionError("Composition escalation must not run for φ_s = 0 ties")

    monkeypatch.setattr(iit3, "_compute_distinctions", explode)
    with config.override(**presets.iit4_2026):
        sia = System(examples.xor_substrate(), (0, 0, 0)).sia()
    assert float(sia.phi) == pytest.approx(0.0)
    assert len(sia.ties) == 2


def test_positive_capped_tie_resolved_by_composition_naturally(noisy_xor_system):
    """The noisy XOR's two φ_s-tied readings support genuinely different
    structures (Φ = 5.55 vs 4.02); Composition selects the congruent
    reading."""
    with config.override(**presets.iit4_2026):
        sia = noisy_xor_system.sia()
    assert float(sia.phi) > 0
    assert len(sia.ties) == 2
    assert tuple(sia.system_state.cause.state) == (0, 0, 0)


def test_positive_phi_tie_extrinsic_keeps_canonical(noisy_xor_system, monkeypatch):
    """φ_s-tied readings whose Φ also ties with intrinsically identical
    structures are an extrinsic tie (S1): the system still qualifies, and a
    canonical representative is reported with the tie surfaced."""
    from pyphi import automorphism
    from pyphi.models.ces import CauseEffectStructure

    monkeypatch.setattr(CauseEffectStructure, "big_phi", property(lambda _self: 1.0))
    monkeypatch.setattr(automorphism, "are_structures_isomorphic", lambda _a, _b: True)
    with config.override(**presets.iit4_2026):
        sia = noisy_xor_system.sia()
    assert float(sia.phi) > 0
    assert len(sia.ties) == 2
    assert tuple(sia.system_state.cause.state) in {(0, 0, 0), (1, 1, 1)}


def test_positive_capped_tie_resolved_by_composition(noisy_xor_system, monkeypatch):
    """When Φ discriminates among φ_s-tied readings, the Φ-maximal reading
    wins (S1: maximal existence via the Composition postulate)."""
    from pyphi.models.ces import CauseEffectStructure

    def fake_big_phi(self):
        cause_state = tuple(self.sia.system_state.cause.state)
        return 5.0 if cause_state == (1, 1, 1) else 1.0

    monkeypatch.setattr(CauseEffectStructure, "big_phi", property(fake_big_phi))
    with config.override(**presets.iit4_2026):
        sia = noisy_xor_system.sia()
    assert float(sia.phi) > 0
    assert tuple(sia.system_state.cause.state) == (1, 1, 1)


def test_positive_phi_tie_nonisomorphic_is_information_violation(
    noisy_xor_system, monkeypatch
):
    """A Φ tie among genuinely distinct structures violates the information
    postulate: the system does not qualify as a complex (S1, strict).

    Forcing the Φ tie exposes the *real* isomorphism check to the two
    readings' genuinely non-isomorphic structures."""
    from pyphi.models.ces import CauseEffectStructure

    monkeypatch.setattr(CauseEffectStructure, "big_phi", property(lambda _self: 1.0))
    with config.override(**presets.iit4_2026):
        sia = noisy_xor_system.sia()
    assert float(sia.phi) == pytest.approx(0.0)
    assert NullResultReason.NONUNIQUE_SYSTEM_STATE in (sia.reasons or [])
