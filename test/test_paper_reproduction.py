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
* **IIT 4.0 (2023), Fig 6C -- "directed cycle".** The 6-unit copy-ring (the only
  Fig 6 panel whose weights are given exactly in the text): ``phi_s = 1.7467``
  (paper 1.74) and ``Phi = 7.65`` (relation sum via analytical relations).
* **IIT 3.0 (2014), Oizumi et al., Fig 12 -- "Assessing the integrated
  conceptual information".** The paper's main worked example: a 3-unit logic-gate
  network (A = OR, B = AND, C = XOR, fully connected; ``pyphi.examples.
  fig4_substrate``) in state (1, 0, 0). The system big-phi reproduces the figure's
  ``Phi = 1.92`` (PyPhi computes 23/12 = 1.9167; the Mayner et al. 2018 practical
  guide reports 1.917 for the same example), and the constellation reproduces the
  six concepts with the published ``phi^Max`` values {0.5, 0.33, 0.25, 0.25, 0.17,
  0.17}.
* **Actual Causation (2019), Albantakis et al., Fig 6 -- "Causal account".** The
  canonical 2-unit OR-AND example (``pyphi.examples.actual_causation_substrate``)
  and the full causal account of the transition {OR, AND} = 10 -> 10: four
  first-order links at ``alpha = log2(4/3) = 0.415`` bits and one second-order
  (joint) cause link at ``alpha = log2(9/8) = 0.170`` bits.

Follow-ons (tracked in ROADMAP N1): the rest of IIT 4.0 Fig 6 (panels A/B/D/E)
and Fig 7 give weights only graphically and need the authors' exact network
definitions; and the Gomez et al. 2020 p53-Mdm2 network.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.convert import le_index2state
from pyphi.direction import Direction
from pyphi.relations import AnalyticalRelations
from pyphi.substrate import Substrate
from pyphi.substrate_generator import build_substrate
from pyphi.substrate_generator import ising
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


# --------------------------------------------------------------------------- #
# IIT 4.0 (2023) -- Fig 6C, "directed cycle"
# --------------------------------------------------------------------------- #
# Of Fig 6's five 6-unit networks, only panel C is specified exactly in the
# text (p32): "a directed cycle in which six units are unidirectionally
# connected with weight w = 1.0 and k = 4. Each unit copies the state of the
# unit before it ... with some indeterminism." The other panels (A, B, D, E)
# give their weights only graphically and are not reliably reconstructable from
# the figure (see ROADMAP N1).
def _fig6c_substrate() -> Substrate:
    """Build the Fig 6C 6-unit directed copy-ring (i copies i-1, w = 1.0)."""
    weights = np.zeros((6, 6))
    for i in range(6):
        weights[i, (i + 1) % 6] = 1.0  # sender -> receiver (ring)
    return build_substrate(
        [ising.probability] * 6,
        weights,
        temperature=1.0 / 4.0,  # ising temperature 1/k reproduces sigmoid(k * .)
        node_labels=tuple("ABCDEF"),
    )


@pytest.mark.slow
def test_iit4_2023_fig6C_copy_ring(_iit4_2023):
    """Fig 6C: the copy-ring's system phi and structure phi (Phi).

    phi_s = 1.7467 (the paper's quoted 1.74 is a two-decimal truncation) and Phi
    -- the summed small-phi of the distinctions and relations -- is 7.65. The
    relation sum is computed analytically (Albantakis et al. 2023, S3) rather
    than by enumerating concrete relations.
    """
    system = System(_fig6c_substrate(), (1, 0, 0, 0, 0, 0), node_indices=tuple(range(6)))
    sia = system.sia()
    assert float(sia.phi) == pytest.approx(1.74, abs=0.01)  # phi_s = 1.7467

    distinctions = system.distinctions().resolve_congruence(sia.system_state)
    sum_phi_d = sum(float(d.phi) for d in distinctions)
    sum_phi_r = float(AnalyticalRelations(distinctions).sum_phi())
    assert round(sum_phi_d + sum_phi_r, 2) == 7.65


# --------------------------------------------------------------------------- #
# IIT 3.0 (2014) -- Oizumi, Albantakis & Tononi, PLoS Comput Biol 10(5):
# e1003588, Fig 12 ("Assessing the integrated conceptual information Phi")
# --------------------------------------------------------------------------- #
# The paper's main worked example (Figs 4, 6, 8-12, 14) is a 3-unit network of
# logic gates -- A = OR, B = AND, C = XOR, fully connected -- evaluated in state
# (1, 0, 0) (the figure's "s_t(ABC) = 100"). This is pyphi.examples.fig4_substrate.
# NOTE: fig4_*system* fixes the *different* state (1, 0, 1); the 2014 worked
# example is in state (1, 0, 0), so we build the substrate and set the state here.
#
# Fig 12 reports the system big-phi Phi^MIP = 1.92 (the MIP is the unidirectional
# cut [A, B] -/-> [C]) and shows the constellation of six concepts with phi^Max
# values {0.5, 0.33, 0.25, 0.25, 0.17, 0.17}. The identical example is worked
# end-to-end in the PyPhi practical guide (Mayner et al. 2018, "Calculating phi"),
# which reports Phi = 1.917 -- the same quantity (23/12 = 1.91666...) to three
# decimals. The two independent published sources agree.
_FIG12_STATE = (1, 0, 0)

# Concept small-phi (phi^Max) by mechanism. Concepts are labeled by mechanism in
# the paper (Figs 9-11); values are quoted to two decimals as in Fig 12. Mechanism
# AC specifies no concept (phi = 0) and is absent from the constellation, leaving
# six concepts.
_FIG12_CONCEPT_PHI = {
    (0,): 0.17,  # A   (1/6)
    (1,): 0.17,  # B   (1/6)
    (2,): 0.25,  # C
    (0, 1): 0.25,  # AB
    (1, 2): 0.33,  # BC  (1/3)
    (0, 1, 2): 0.50,  # ABC
}


@pytest.fixture
def _iit3():
    with config.override(
        **presets.iit3, validate_system_states=False, progress_bars=False
    ):
        yield


def test_iit3_2014_fig12_system_phi(_iit3):
    """Fig 12: the worked example's system integrated information Phi = 1.92.

    PyPhi computes 23/12 = 1.91666..., which rounds to the paper's 1.92 and
    matches the 1.917 reported by the Mayner et al. 2018 practical guide.
    """
    sia = System(examples.fig4_substrate(), _FIG12_STATE, node_indices=(0, 1, 2)).sia()
    assert round(float(sia.phi), 2) == 1.92
    assert float(sia.phi) == pytest.approx(23 / 12, abs=1e-4)


def test_iit3_2014_fig12_constellation(_iit3):
    """Fig 12: the constellation of six concepts with their published phi^Max.

    Checks the concept count, the phi^Max multiset to the figure's two decimals,
    and the per-mechanism assignment (concepts are labeled by mechanism in the
    paper's Figs 9-11).
    """
    ces = System(examples.fig4_substrate(), _FIG12_STATE, node_indices=(0, 1, 2)).ces()
    concepts = list(ces)
    assert len(concepts) == 6
    assert sorted(round(float(c.phi), 2) for c in concepts) == sorted(
        _FIG12_CONCEPT_PHI.values()
    )
    by_mechanism = {tuple(c.mechanism): c for c in concepts}
    assert set(by_mechanism) == set(_FIG12_CONCEPT_PHI)
    for mechanism, phi in _FIG12_CONCEPT_PHI.items():
        assert round(float(by_mechanism[mechanism].phi), 2) == phi


# --------------------------------------------------------------------------- #
# Actual Causation (2019) -- Albantakis, Marshall, Hoel & Tononi,
# Entropy 21(5):459, Fig 6 ("Causal account")
# --------------------------------------------------------------------------- #
# The paper's canonical actual-causation example (Figs 2/3/6) is a 2-unit
# substrate -- an OR gate and an AND gate, each with a self-loop and a reciprocal
# connection (pyphi.examples.actual_causation_substrate). Fig 6 gives the full
# causal account of the transition {OR, AND} = 10 -> 10 (before = after =
# (1, 0)): each first-order link (OR and AND, as both cause and effect) has
# alpha = 0.415 bits = log2(4/3), and the single second-order link -- the joint
# cause {OR, AND} = 10 -< {OR, AND} = 10 -- has alpha = 0.170 bits = log2(9/8).
# (There is no irreducible second-order *effect* link, so the joint appears once.)
#
# NOTE on figure number: the roadmap's "AC 2019 Fig 11" is the 7-unit "voting"
# example, whose weights are given only graphically and which is not pre-built in
# pyphi.examples; Fig 6 is the canonical example pyphi.examples provides.
_AC_FIG6_BEFORE = (1, 0)
_AC_FIG6_AFTER = (1, 0)

# (direction, mechanism) -> (purview, alpha-in-bits). Alpha is quoted to the
# paper's three decimals; PyPhi computes log2(4/3) = 0.415037 and
# log2(9/8) = 0.169925.
_AC_FIG6_ACCOUNT = {
    (Direction.CAUSE, (0,)): ((0,), 0.415),  # OR  -< OR
    (Direction.CAUSE, (1,)): ((1,), 0.415),  # AND -< AND
    (Direction.CAUSE, (0, 1)): ((0, 1), 0.170),  # {OR, AND} -< {OR, AND}
    (Direction.EFFECT, (0,)): ((0,), 0.415),  # OR  >- OR
    (Direction.EFFECT, (1,)): ((1,), 0.415),  # AND >- AND
}


def test_ac_2019_fig6_or_and_account(_iit3):
    """AC Fig 6: the full causal account of the OR-AND transition 10 -> 10.

    Four first-order links at alpha = log2(4/3) = 0.415 bits and one
    second-order (joint) cause link at alpha = log2(9/8) = 0.170 bits, matching
    the paper's reported account exactly.
    """
    transition = actual.Transition(
        examples.actual_causation_substrate(),
        _AC_FIG6_BEFORE,
        _AC_FIG6_AFTER,
        (0, 1),
        (0, 1),
    )
    account = {
        (link.direction, tuple(link.mechanism)): (
            tuple(link.purview),
            round(float(link.alpha), 3),
        )
        for link in actual.account(transition)
    }
    assert account == _AC_FIG6_ACCOUNT
