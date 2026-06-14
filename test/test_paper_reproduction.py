"""N1 -- paper-reproduction acceptance suite.

This suite reproduces *published* worked examples from the IIT literature with
expected values taken from the papers themselves (not from PyPhi), wired in as a
CI gate so a regression that silently diverges from the published science fails
loudly. Each entry documents its source figure/table and the provenance of every
pinned number.

Unlike ``test_golden_regression.py`` (PyPhi-self-referential numerical pins that
lock the *current* output), the assertions here are **paper-sourced**: they pass
only if PyPhi agrees with the published result.

Currently covered
-----------------
* **IIT 4.0 (2023), Albantakis et al., Fig 1 -- "Identifying substrates of
  consciousness".** The logistic substrate of Fig 1A in state ``aBC`` and the
  system integrated information ``phi_s`` of its candidate systems (Fig 1E).
  Five published values reproduce to two decimals: ``phi_s`` for a / aB / aBC
  (Fig 1E) and the cause/effect split ``phi_c = 0.24`` / ``phi_e = 0.17`` of aB
  (Fig 1D). Cross-checked against paper-era PyPhi 1.2.0 (commit 75d0c411): both
  versions agree that the single unit {C} has the highest ``phi_s`` here
  (~0.21-0.29), so aB is *a* complex (maximal among the systems overlapping it),
  not the global maximum -- exactly the paper's claim. Current PyPhi is the
  paper-faithful one: its ``DIRECTED_SET_PARTITION`` scheme reproduces
  ``phi_s(a) = 0.04`` where the old ``SET_UNI/BI`` scheme gives 0.068.
* **IIT 4.0 (2023), Fig 2 -- "Composition and causal distinctions".** The three
  irreducible distinctions of complex aB, with ``phi_d(a) = 0.33``,
  ``phi_d(B) = 0.32``, ``phi_d(aB) = 0.07`` and their cause/effect purviews.
* **IIT 4.0 (2023), Fig 4 -- "Composition and causal relations".** The relation
  ``r({a, aB})`` over shared purview unit b: ``phi_r = 0.035`` and 9 faces.

Follow-ons (tracked in ROADMAP N1): Fig 6/7 (larger logistic substrates); the
IIT 3.0 (2014) Fig 1 example; AC 2019 Fig 11; and the Gomez et al. 2020
p53-Mdm2 network.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyphi.conf import config
from pyphi.conf import presets
from pyphi.convert import le_index2state
from pyphi.substrate import Substrate
from pyphi.system import System

# --------------------------------------------------------------------------- #
# IIT 4.0 (2023) -- Albantakis et al., PLoS Comput Biol 19(10): e1011465, Fig 1
# --------------------------------------------------------------------------- #
# Fig 1A defines a 3-unit logistic substrate (units A, B, C). The activation
# function (paper Eq 60) is a sigmoid of the weighted inputs in {-1, +1}:
#
#     p(unit_j = ON) = 1 / (1 + exp(-k * sum_i w[i, j] * s_i)),   s_i in {-1, +1}
#
# with slope k = 4.0. The connection weights are read from the Fig 1A causal
# model (black = excitatory, orange = inhibitory; the dot marks the inhibited
# unit):
#
#     A->A = -0.2   A->B = +0.7   A->C = +0.2
#     B->A = +0.7   B->B = -0.2   (no B->C)
#     (no C->A)     C->B = -0.8   C->C = +0.2
#
# This reading is self-validating: the three published phi_s values of Fig 1E
# (below) all reproduce to the paper's two-decimal precision, which they would
# not if any weight were misread.
_FIG1_K = 4.0
_FIG1_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
# Fig 1A: the substrate is shown in state aBC = (a off, B on, C on). Lowercase
# denotes state "-1" (PyPhi 0), uppercase "+1" (PyPhi 1).
_FIG1_STATE = (0, 1, 1)

# Fig 1E ("Exclusion"): published system integrated information phi_s for three
# candidate systems. aB is a complex -- its phi_s exceeds that of every system
# overlapping it, here illustrated by its subset a and its superset aBC. Values
# are quoted to two decimals in the paper.
_FIG1_PUBLISHED_PHI_S = {
    (0,): 0.04,  # subset a
    (0, 1): 0.17,  # complex aB
    (0, 1, 2): 0.13,  # superset aBC
}

# Fig 2 ("Composition and causal distinctions"): the irreducible distinctions
# D(aB) -- two first-order (a, B) and one second-order (aB) -- with their
# small-phi (phi_d) and cause/effect purview units. The paper labels purviews by
# their specified *state* (e.g. cause "b", effect "Ab"); we validate the purview
# unit sets (node indices) together with phi_d (quoted to two decimals).
_FIG2_DISTINCTIONS = {
    # mechanism: (phi_d, cause_purview, effect_purview)
    (0,): (0.33, (1,), (1,)),  # d(a): cause b, effect b
    (1,): (0.32, (0,), (0, 1)),  # d(B): cause A, effect Ab
    (0, 1): (0.07, (1,), (0, 1)),  # d(aB): cause b, effect Ab
}


def _fig1_substrate() -> Substrate:
    """Build the IIT 4.0 (2023) Fig 1A logistic substrate."""
    n = _FIG1_WEIGHTS.shape[0]
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = np.array([2 * b - 1 for b in le_index2state(row, n)])  # {0,1} -> {-1,+1}
        for j in range(n):
            net_input = float(_FIG1_WEIGHTS[:, j] @ s)
            tpm[row, j] = 1.0 / (1.0 + np.exp(-_FIG1_K * net_input))
    cm = (_FIG1_WEIGHTS != 0).astype(int)
    return Substrate(tpm, cm=cm, node_labels=("A", "B", "C"))


@pytest.fixture
def _iit4_2023():
    with config.override(
        **presets.iit4_2023, validate_system_states=False, progress_bars=False
    ):
        yield


@pytest.mark.parametrize(("candidate", "expected"), list(_FIG1_PUBLISHED_PHI_S.items()))
def test_iit4_2023_fig1_system_phi(_iit4_2023, candidate, expected):
    """Reproduce the Fig 1E published phi_s for each candidate system."""
    system = System(_fig1_substrate(), _FIG1_STATE, node_indices=candidate)
    phi_s = float(system.sia().phi)
    assert round(phi_s, 2) == expected, (
        f"candidate {candidate}: phi_s={phi_s:.4f} rounds to {round(phi_s, 2)}, "
        f"paper Fig 1E reports {expected}"
    )


def test_iit4_2023_fig1_aB_is_a_complex(_iit4_2023):
    """Fig 1's exclusion result: aB is a complex.

    The paper defines a complex as a system whose ``phi_s`` exceeds that of all
    *overlapping* systems (those sharing at least one unit with it). aB = {A, B}
    overlaps every candidate except {C}; its phi_s (0.17) must strictly exceed
    each. (PyPhi's *global* maximal complex here is the non-overlapping {C} with
    phi_s ~ 0.21 -- consistent with the paper, which presents aB as *a* complex,
    not the global maximum.)
    """
    substrate = _fig1_substrate()
    aB = (0, 1)
    phi_aB = float(System(substrate, _FIG1_STATE, node_indices=aB).sia().phi)
    assert round(phi_aB, 2) == 0.17
    for size in range(1, 4):
        for candidate in itertools.combinations(range(3), size):
            if candidate == aB or set(candidate).isdisjoint(aB):
                continue  # skip aB itself and non-overlapping systems
            phi = float(System(substrate, _FIG1_STATE, node_indices=candidate).sia().phi)
            assert phi < phi_aB, (
                f"overlapping system {candidate} has phi_s={phi:.4f} >= aB's "
                f"{phi_aB:.4f}; aB would not be a complex"
            )


def test_iit4_2023_fig1D_aB_cause_effect_phi(_iit4_2023):
    """Fig 1D ("Integration"): aB's phi_s = min(phi_c, phi_e), with the published
    cause/effect split phi_c = 0.24 and phi_e = 0.17."""
    sia = System(_fig1_substrate(), _FIG1_STATE, node_indices=(0, 1)).sia()
    assert round(float(sia.cause.phi), 2) == 0.24
    assert round(float(sia.effect.phi), 2) == 0.17
    assert round(float(sia.phi), 2) == 0.17  # phi_s = min(phi_c, phi_e)


def test_iit4_2023_fig2_distinctions(_iit4_2023):
    """Fig 2: the irreducible distinctions D(aB) and their published phi_d and
    cause/effect purviews."""
    ces = System(_fig1_substrate(), _FIG1_STATE, node_indices=(0, 1)).ces()
    by_mechanism = {tuple(d.mechanism): d for d in ces.distinctions}
    assert set(by_mechanism) == set(_FIG2_DISTINCTIONS)
    for mechanism, (phi_d, cause_pv, effect_pv) in _FIG2_DISTINCTIONS.items():
        distinction = by_mechanism[mechanism]
        assert round(float(distinction.phi), 2) == phi_d
        assert tuple(distinction.cause.purview) == cause_pv
        assert tuple(distinction.effect.purview) == effect_pv


def test_iit4_2023_fig4_relation_a_aB(_iit4_2023):
    """Fig 4 ("Composition and causal relations"): the relation r({a, aB}) binds
    distinctions a and aB over their shared purview unit b, forming all 9 faces.

    PyPhi computes phi_r = 0.0357 = phi_d(aB) / 2 -- the relation reduces to the
    aB distinction's phi spread over its two-unit purview union (the binding
    minimum). The paper's reported 0.035 is the same quantity computed from the
    rounded phi_d(aB) = 0.07 (0.07 / 2 = 0.035).
    """
    ces = System(_fig1_substrate(), _FIG1_STATE, node_indices=(0, 1)).ces()
    relation = next(
        r for r in ces.relations if {tuple(m) for m in r.mechanisms} == {(0,), (0, 1)}
    )
    assert relation.num_faces == 9
    assert float(relation.phi) == pytest.approx(0.035, abs=1e-3)
