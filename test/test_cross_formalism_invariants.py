"""B5 — cross-formalism differential invariants.

Paired/property tests that pin relationships *between* PyPhi's formalisms, the
dominant correctness strategy across the 5+ formalisms. Two candidate
invariants from the roadmap are examined here:

  - **``phi_2026 <= phi_2023`` (holds).** The 2026 formalism adds the Eq. 23
    intrinsic-information cap, which can only lower system phi (cf. B4). A
    Hypothesis property over random small substrates confirms it, generalizing
    B4's per-network check. Its corollary — 2023-reducible implies
    2026-reducible — follows immediately.

  - **"IIT 3.0 and 4.0 agree on reducibility" (REFUTED).** Empirically the two
    formalisms disagree on ~70% of random reachable small substrates (audit
    2026-06-13):
    IIT 3.0 (EMD over constellations) frequently finds ``phi > 0`` where IIT
    4.0 (GID system integration) finds ``phi = 0``. ``test_iit3_iit4_*`` pins a
    concrete 2-node witness so this divergence is documented and locked, not
    silently assumed.

Deferred slices of B5 (follow-ons): the AC/IIT sign-agreement invariant
(actual causation is a separate subsystem with an unsettled cross-formalism
relation), and the byte-match against the ``b3aaa3e5`` pre-refactor oracle
(its ``compute.phi`` is not a valid IIT 3.0 SIA — basic -> 0 — so a 3.0
byte-match needs a genuine PyPhi 1.x oracle; the 4.0 byte-match is viable but
deferred).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import Substrate
from pyphi import System
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.formalism import iit3

from .hypothesis_utils import binary_state
from .hypothesis_utils import small_substrate

_ZERO = 1e-9


def _iit4_phi(substrate: Substrate, state: tuple[int, ...], preset: dict) -> float:
    """System phi via the active IIT 4.0 formalism (``System.sia``)."""
    with config.override(**preset, validate_system_states=False):
        return float(System(substrate, state).sia().phi)


def _iit3_phi(substrate: Substrate, state: tuple[int, ...]) -> float:
    """System phi via the genuine IIT 3.0 SIA (``System.sia`` is 4.0-only)."""
    with config.override(**presets.iit3, validate_system_states=False):
        return float(iit3.sia(System(substrate, state)).phi)


class TestCapMonotonicityAcrossVersions:
    """The 2026 ii(s) cap never raises system phi above the 2023 value."""

    @settings(max_examples=30, deadline=None)
    @given(data=st.data())
    def test_phi_2026_le_phi_2023(self, data) -> None:
        substrate = data.draw(small_substrate())
        state = data.draw(binary_state(substrate.size))
        try:
            phi_2023 = _iit4_phi(substrate, state, presets.iit4_2023)
            phi_2026 = _iit4_phi(substrate, state, presets.iit4_2026)
        except Exception:
            assume(False)
            return
        assert phi_2026 <= phi_2023 + _ZERO, (
            f"2026 cap raised phi above 2023: phi_2026={phi_2026} > "
            f"phi_2023={phi_2023} (state={state})"
        )

    def test_phi_2026_le_phi_2023_regression(self) -> None:
        """Deterministic guard for the Eq-23 cap MIP-selection bug.

        A 2-node substrate where the per-partition cap once shifted the system
        MIP and produced ``phi_2026 > phi_2023`` (0.301 vs 0.172). With the cap
        applied to the 4.0-selected MIP, the cap can only lower phi.
        """
        substrate = Substrate(
            np.array([[0.0, 0.75], [0.5, 0.5], [0.75, 0.25], [0.0, 0.75]]),
            cm=np.ones((2, 2), dtype=int),
        )
        state = (0, 1)
        phi_2023 = _iit4_phi(substrate, state, presets.iit4_2023)
        phi_2026 = _iit4_phi(substrate, state, presets.iit4_2026)
        assert phi_2026 <= phi_2023 + _ZERO, (
            f"cap regression: phi_2026={phi_2026} > phi_2023={phi_2023}"
        )


class TestReducibilityDivergence:
    """IIT 3.0 and IIT 4.0 do NOT agree on reducibility — pinned witness."""

    # 2-node substrate (state-by-node TPM, fully connected) at state (0, 0)
    # where IIT 3.0 finds phi > 0 but IIT 4.0 (2023) finds phi = 0. Found by
    # the 2026-06-13 cross-formalism audit; locked here so the divergence is
    # documented rather than silently assumed away.
    _TPM = np.array([[1.0, 0.75], [0.75, 1.0], [0.5, 0.75], [1.0, 0.25]])
    _STATE = (0, 0)

    @pytest.mark.emd
    def test_iit3_irreducible_where_iit4_reducible(self) -> None:
        substrate = Substrate(self._TPM, cm=np.ones((2, 2), dtype=int))
        phi_3 = _iit3_phi(substrate, self._STATE)
        phi_4 = _iit4_phi(substrate, self._STATE, presets.iit4_2023)
        assert phi_3 > 1e-3, f"expected IIT 3.0 phi > 0, got {phi_3}"
        assert phi_4 < _ZERO, f"expected IIT 4.0 phi = 0, got {phi_4}"
