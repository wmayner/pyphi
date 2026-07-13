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
from pyphi import numerics
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


def _independent_i_diff(
    system: System, direction: Direction, specified_state: tuple[int, ...]
) -> float:
    """Re-derive i_diff from the paper's definitions, independent of the
    production measure's conventions.

    Mayner et al. (2026): Eq. 4, ``i_diff_e(s, s') = -log2 p_e(s' | s)``;
    Eqs. 6 + 11, ``i_diff_c(s, s') = -log2 p_c_arrow(s' | s)`` with the
    Bayes-normalized cause posterior
    ``p_c_arrow(sbar | s) = p_c(s | sbar) / sum_shat p_c(s | shat)``. Both
    are evaluated at the *specified* state s' (Eq. 12), not the current
    state.
    """
    nodes = system.node_indices
    rep = np.asarray(system.forward_repertoire(direction, nodes, nodes)).squeeze()
    if direction == Direction.CAUSE:
        # The forward cause array holds the likelihoods p_c(s | sbar);
        # Eq. 11 normalizes them into the posterior over cause states.
        rep = rep / rep.sum()
    p = float(rep[tuple(specified_state)])
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
            specified = tuple(sia.system_state[direction].state)
            independent = _independent_i_diff(system, direction, specified)
            assert numerics.eq(production, independent), (
                f"{name} {direction.name}: production i_diff {production} != "
                f"paper-formula i_diff at specified state {independent}"
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
    assert numerics.eq(recomposed, float(sia26.phi)), (
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


##############################################################################
# Paper-pinned monad values (Mayner et al. 2026, Example 1 / Fig. 2)
##############################################################################


def _monad(p_on_from_off: float, p_on_from_on: float) -> Substrate:
    """Single-unit substrate with the given P(ON at t+1 | state at t)."""
    tpm = np.array([[p_on_from_off], [p_on_from_on]])
    return Substrate(tpm, cm=np.array([[1]]), node_labels=["m"])


def _paper_monad_phi(p_on_from_off: float, p_on_from_on: float, s: int) -> float:
    """φ_s = ii(s) for a monad (Eq. 14), from Eqs. 4-13 in pure numpy.

    Independent of PyPhi's repertoire machinery: works directly on the
    transition probabilities.
    """
    p_on = np.array([p_on_from_off, p_on_from_on])

    # Effect side: p_e(sbar | s) and the unconstrained p_e(sbar) (Eq. 8).
    p_e_cond = np.array([1.0 - p_on[s], p_on[s]])
    p_e_unc = np.array([1.0 - p_on.mean(), p_on.mean()])
    i_spec_e = p_e_cond * np.log2(p_e_cond / p_e_unc)  # Eq. 7
    sprime_e = int(np.argmax(i_spec_e))  # Eq. 12
    ii_e = min(i_spec_e[sprime_e], -np.log2(p_e_cond[sprime_e]))  # Eqs. 4, 13

    # Cause side: likelihoods p_c(s | sbar), Bayes posterior (Eq. 11),
    # unconditional p_c(s) (Eq. 10).
    likelihood = np.array([[1.0 - p_on[0], p_on[0]], [1.0 - p_on[1], p_on[1]]])[
        :, s
    ]  # p(current = s | prior = sbar), for sbar in (0, 1)
    posterior = likelihood / likelihood.sum()
    p_c_unc = likelihood.mean()
    i_spec_c = posterior * np.log2(likelihood / p_c_unc)  # Eq. 9
    sprime_c = int(np.argmax(i_spec_c))  # Eq. 12
    ii_c = min(i_spec_c[sprime_c], -np.log2(posterior[sprime_c]))  # Eqs. 6, 13

    return float(min(ii_c, ii_e))  # Eq. 13; monad: φ_s = ii(s) (Eq. 14)


@pytest.mark.parametrize(
    ("p_on_from_off", "p_on_from_on", "regime"),
    [
        # Paper Fig. 2 monad at p=0.9. NOT regime: specified state s' != s,
        # the case the paper's symmetry claim covers ("the symmetric case
        # corresponds to an imperfect NOT gate"). Eq. 27: φ_s = -log2(0.9).
        (0.9, 0.1, "not"),
        # COPY regime: s' == s; conventions agree (regression pin).
        (0.1, 0.9, "copy"),
        # Asymmetric: non-doubly-stochastic, so the Eq. 11 cause normalizer
        # differs from 1 and the cause-side i_diff exercises it.
        (0.9, 0.15, "asymmetric"),
    ],
)
def test_monad_phi_matches_paper_formulas(p_on_from_off, p_on_from_on, regime):
    """The monad φ_s equals the from-scratch Eqs. 4-14 value."""
    expected = _paper_monad_phi(p_on_from_off, p_on_from_on, s=1)
    with config.override(
        **presets.iit4_2026, single_micro_nodes_with_selfloops_have_phi=True
    ):
        sia = System(_monad(p_on_from_off, p_on_from_on), (1,)).sia()
    assert float(sia.phi) == pytest.approx(expected, abs=1e-10), (
        f"{regime}: PyPhi φ_s {float(sia.phi)} != paper Eqs. 4-14 value {expected}"
    )


def test_monad_maximum_matches_paper_fig2c():
    """The paper prints max φ_s = 0.427 at p = 0.744 (Fig. 2C)."""
    p = 0.744
    with config.override(
        **presets.iit4_2026, single_micro_nodes_with_selfloops_have_phi=True
    ):
        sia = System(_monad(1.0 - p, p), (1,)).sia()
    assert round(float(sia.phi), 3) == pytest.approx(0.427)


def test_cap_applies_to_all_tie_members():
    """Every member of ``sia.ties`` carries its own capped φ.

    The XOR loop is deterministic, so i_diff = 0 in both directions and the
    cap forces φ_s = 0 for every tied (cause, effect) state pair — not just
    the selected one.
    """
    from pyphi import examples

    with config.override(**presets.iit4_2026):
        sia = System(examples.xor_substrate(), (0, 0, 0)).sia()
    assert float(sia.phi) == pytest.approx(0.0)
    for tied in sia.ties:
        assert float(tied.phi) == pytest.approx(0.0), (
            f"tie member carries uncapped φ = {float(tied.phi)}"
        )
