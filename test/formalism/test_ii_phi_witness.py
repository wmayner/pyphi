"""The min(ii_c, ii_e) ≥ φ_s inequality: a refuting witness under 2023/GID.

The conjecture that a system's integrated information is bounded by its
intrinsic information, ``min(ii_c, ii_e) ≥ φ_s``, is **false** under the IIT
4.0 (2023) formalism with the generalized-intrinsic-difference system measure.
This module pins the minimal counterexample and confirms that the 2026 Eq. 23
cap removes the violation by construction.

The witness is a two-unit substrate found by random search over asymmetric
transition probability matrices; see
``experiments/ii_phi_inequality_experiments/FINDINGS.md`` for the derivation and the raw
data (``hunt_random_seed20260708.json.gz``). Its margin is three orders of
magnitude beyond ``config.numerics.precision``, so the violation is not a
tie-resolution artifact.
"""

import numpy as np
import pytest

import pyphi
from pyphi import Substrate
from pyphi import numerics
from pyphi.conf import presets
from pyphi.system import System

# Full-precision witness from ``hunt_random_seed20260708.json.gz`` (the minimal
# violation, margin −0.054). State-by-node, rows little-endian over (n0, n1),
# columns P(unit = 1); complete connectivity; observed state (0, 1).
WITNESS_TPM = np.array(
    [
        [0.9029022616191246, 0.7445944260398637],
        [0.5550292646159541, 0.24026118456656692],
        [0.5431971066061434, 0.4245605122716022],
        [0.07088262072300057, 0.8440683613857206],
    ]
)
WITNESS_STATE = (0, 1)


def _witness_system():
    substrate = Substrate(WITNESS_TPM, cm=np.ones((2, 2)), node_labels=["n0", "n1"])
    return System(substrate, state=WITNESS_STATE, node_indices=(0, 1))


def _phi_and_ii(preset) -> tuple[float, float, float]:
    with pyphi.config.override(progress_bars=False, **preset):
        sia = _witness_system().sia()
        phi = float(sia.phi)
        ii_cause = float(sia.system_state.cause.intrinsic_information)
        ii_effect = float(sia.system_state.effect.intrinsic_information)
    return phi, ii_cause, ii_effect


def test_gid_violates_ii_phi_inequality():
    # Under 2023/GID the system is more integrated than it is informative:
    # φ_s exceeds min(ii_c, ii_e), refuting the inequality.
    phi, ii_cause, ii_effect = _phi_and_ii(presets.iit4_2023)
    assert phi == pytest.approx(0.17192794, abs=1e-6)
    assert ii_effect == pytest.approx(0.11784697, abs=1e-6)
    margin = min(ii_cause, ii_effect) - phi
    assert margin == pytest.approx(-0.05408097, abs=1e-6)
    assert margin < 0 and not numerics.is_zero(margin)  # beyond tolerance


def test_2026_cap_restores_the_inequality():
    # The 2026 Eq. 23 cap sets φ_s = min over directions of ii, so
    # min(ii_c, ii_e) ≥ φ_s holds by construction and the cap strictly binds
    # (capped φ_s < the 2023 φ_s of 0.172), making this a minimal cap-biting
    # substrate with positive φ.
    phi, ii_cause, ii_effect = _phi_and_ii(presets.iit4_2026)
    assert phi == pytest.approx(ii_effect)
    assert min(ii_cause, ii_effect) >= phi
    assert phi < 0.17192794  # the cap strictly reduces φ_s
    assert phi > 0  # ... but does not zero it
