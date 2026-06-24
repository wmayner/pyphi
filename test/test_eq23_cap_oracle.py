"""B4 — differential oracle for the IIT 4.0 (2026) intrinsic-information cap.

The 2026 formalism caps system phi by the intrinsic-information requirement
(Eq. 23): ``phi_s = min{phi_c, phi_e, ii(s)}`` with
``ii(s) = min_d min(i_spec_d, i_diff_d)``. Production applies this in
``formalism.iit4.evaluate_partition``; the cap's correctness was otherwise
checked only by the same code that computes it. This module is an independent
cross-check that:

  1. re-derives ``i_diff_d = -log2 P_forward(proper_state)`` from scratch and
     confirms it equals the value production stores on the SIA;
  2. confirms the cap-composition identity — because the cap terms are
     partition-independent, the 2026 MIP satisfies
     ``phi_2026 = |min(phi_2023, i_spec_c, i_diff_c, i_spec_e, i_diff_e)|+``;
  3. pins that the cap *strictly binds* with non-zero phi on a constructed
     network (``logistic3_k8``) — the regime that exercises the ``min`` at a
     non-edge point — and that ``phi_2026 <= phi_2023`` everywhere.

(The GID ``i_spec`` term is cross-checked structurally via the composition
identity; a from-scratch re-derivation of the specification measure is a
deeper follow-up.)
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi import utils
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction


def _logistic3_k8() -> Substrate:
    """3-node fully-connected logistic substrate (k=8, weights 0.3).

    Barely stochastic: all distinctions have positive i_diff while phi exceeds
    the smallest of them, so the 2026 ii(s) cap binds at a non-trivial
    intermediate value (phi_2023 ~ 0.037, phi_2026 ~ 0.003).
    """
    k = 8.0
    weights = np.full((3, 3), 0.3)
    cm = np.ones((3, 3), dtype=int)
    tpm = np.zeros((8, 3))
    for i, s in enumerate(itertools.product([-1, 1], repeat=3)):
        for j in range(3):
            inp = sum(weights[ki, j] * s[ki] for ki in range(3))
            tpm[i, j] = 1.0 / (1.0 + np.exp(-k * inp))
    return Substrate(tpm, cm)


def _independent_i_diff(system: System, direction: Direction) -> float:
    """Re-derive i_diff = min positive -log2(P) of the unpartitioned forward
    repertoire at the system's actual state, independent of the production
    ``intrinsic_differentiation`` measure."""
    nodes = system.node_indices
    rep = np.asarray(system.forward_repertoire(direction, nodes, nodes)).squeeze()
    p = float(rep[system.proper_state])
    return -np.log2(p) if 0.0 < p < 1.0 else 0.0


def _sia_pair(substrate: Substrate, state: tuple[int, ...]):
    """Return (sia_2023, sia_2026) for a substrate/state."""
    with config.override(**presets.iit4_2023):
        sia23 = System(substrate, state).sia()
    with config.override(**presets.iit4_2026):
        sia26 = System(substrate, state).sia()
    return sia23, sia26


# State per substrate. logistic3_k8 binds the cap; basic/xor are non-binding
# (the cap is a no-op there), so the identity must hold in both regimes.
_STATES = {
    "logistic3_k8": (0, 0, 0),
    "basic": (1, 0, 0),
    "xor": (0, 0, 0),
}


def _substrate(name: str) -> Substrate:
    from pyphi import examples

    return {
        "logistic3_k8": _logistic3_k8,
        "basic": examples.basic_substrate,
        "xor": examples.xor_substrate,
    }[name]()


@pytest.mark.parametrize("name", ["logistic3_k8", "basic", "xor"])
def test_independent_i_diff_matches_production(name: str) -> None:
    """The from-scratch i_diff = -log2 P(state) equals production's stored value."""
    substrate, state = _substrate(name), _STATES[name]
    with config.override(**presets.iit4_2026):
        sia = System(substrate, state).sia()
        system = System(substrate, state)
        for direction in (Direction.CAUSE, Direction.EFFECT):
            production = float(sia.intrinsic_differentiation[direction])
            independent = _independent_i_diff(system, direction)
            assert utils.eq(production, independent), (
                f"{name} {direction.name}: production i_diff {production} != "
                f"independent -log2 P(state) {independent}"
            )


@pytest.mark.parametrize("name", ["logistic3_k8", "basic", "xor"])
def test_cap_composition_identity(name: str) -> None:
    """phi_2026 == |min(phi_2023, i_spec_c, i_diff_c, i_spec_e, i_diff_e)|+.

    Independently recomposes the Eq. 23 cap from the uncapped 2023 phi and the
    (partition-independent) cap terms, and confirms it reproduces production's
    2026 phi exactly.
    """
    substrate, state = _substrate(name), _STATES[name]
    sia23, sia26 = _sia_pair(substrate, state)
    phi_2023 = float(sia23.phi)
    terms = [phi_2023]
    for direction in (Direction.CAUSE, Direction.EFFECT):
        terms.append(float(sia26.system_state[direction].intrinsic_information))
        terms.append(float(sia26.intrinsic_differentiation[direction]))
    recomposed = utils.positive_part(min(terms))
    assert utils.eq(recomposed, float(sia26.phi)), (
        f"{name}: cap composition {recomposed} != production phi_2026 "
        f"{float(sia26.phi)} (terms={[round(t, 4) for t in terms]})"
    )


def test_cap_strictly_binds_with_nonzero_phi() -> None:
    """logistic3_k8 is a cap-biting network: 0 < phi_2026 < phi_2023."""
    sia23, sia26 = _sia_pair(_logistic3_k8(), (0, 0, 0))
    phi_2023, phi_2026 = float(sia23.phi), float(sia26.phi)
    assert phi_2026 > 1e-9, f"expected phi_2026 > 0, got {phi_2026}"
    assert phi_2026 < phi_2023 - 1e-9, (
        f"expected the cap to strictly bind (phi_2026 < phi_2023), got "
        f"phi_2026={phi_2026} phi_2023={phi_2023}"
    )


@pytest.mark.parametrize("name", ["logistic3_k8", "basic", "xor"])
def test_cap_never_increases_phi(name: str) -> None:
    """The 2026 cap can only lower phi: phi_2026 <= phi_2023."""
    sia23, sia26 = _sia_pair(_substrate(name), _STATES[name])
    assert float(sia26.phi) <= float(sia23.phi) + 1e-9
