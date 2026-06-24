"""Tests for Differentiation (cross-structure projection; D and D_p)."""

import pytest

from pyphi import examples
from pyphi.matching import Differentiation
from pyphi.matching import PerceptualSystem
from pyphi.matching.perception import Perception

# --- Hand-computed checks on duck-typed stand-ins ---------------------------
# The grid3 fixture's two structures share no components, so the
# max-over-structures path is anchored here with controlled overlap.


class FakeComponent:
    """Hashable stand-in for a distinction or relation with a phi value."""

    def __init__(self, name, phi):
        self.name = name
        self.phi = phi

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class FakeCES:
    def __init__(self, distinctions, relations):
        self.distinctions = distinctions
        self.relations = relations


class FakePerception:
    """Duck-typed Perception with fixed per-component perception values."""

    def __init__(self, distinctions, relations, values):
        self.ces = FakeCES(distinctions, relations)
        self._values = values

    def distinction_perception(self, distinction):
        return self._values[distinction]

    def relation_perception(self, relation):
        return self._values[relation]


A = FakeComponent("a", phi=1.0)
B = FakeComponent("b", phi=2.0)
R = FakeComponent("r", phi=0.5)


def _fake_structures():
    s1 = FakePerception([A, B], [R], {A: 0.3, B: 0.8, R: 0.1})
    s2 = FakePerception([A], [R], {A: 0.5, R: 0.05})
    return (s1, s2)


def test_differentiation_hand_computed():
    d = Differentiation(_fake_structures())  # pyright: ignore[reportArgumentType]
    # Unique components: a, b, r -> D = 1.0 + 2.0 + 0.5
    assert d.differentiation == pytest.approx(3.5)


def test_perceptual_differentiation_hand_computed():
    d = Differentiation(_fake_structures())  # pyright: ignore[reportArgumentType]
    # Max perception per component: a -> 0.5, b -> 0.8, r -> 0.1
    assert d.perceptual_differentiation == pytest.approx(1.4)


def test_projection_hand_computed():
    d = Differentiation(_fake_structures())  # pyright: ignore[reportArgumentType]
    assert d.projection == {A: 0.5, B: 0.8, R: 0.1}


def test_empty_differentiation_is_zero():
    d = Differentiation(())
    assert d.differentiation == 0.0
    assert d.perceptual_differentiation == 0.0
    assert d.projection == {}


# --- Invariants on real Perceptions (grid3 fixture) -------------------------


def _full_state(sensory_indices, system_indices, x, y):
    n = len(sensory_indices) + len(system_indices)
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, yi in zip(system_indices, y, strict=True):
        full[i] = yi
    return tuple(full)


@pytest.fixture(scope="module")
def perceptions():
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    result = {}
    for stimulus in [(0,), (1,)]:
        y = ttpm.argmax_state(stimulus)
        ces = substrate.ces(
            state=_full_state(sensory, system, stimulus, y), indices=system
        )
        result[stimulus] = Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)
    return result


def test_single_structure_differentiation_is_big_phi(perceptions):
    p = perceptions[(0,)]
    d = Differentiation((p,))
    assert d.differentiation == pytest.approx(float(p.ces.big_phi))


def test_single_structure_perceptual_differentiation_is_richness(perceptions):
    p = perceptions[(0,)]
    d = Differentiation((p,))
    assert d.perceptual_differentiation == pytest.approx(p.richness)


def test_duplicate_structure_is_idempotent(perceptions):
    p = perceptions[(0,)]
    once = Differentiation((p,))
    twice = Differentiation((p, p))
    assert twice.differentiation == pytest.approx(once.differentiation)
    assert twice.perceptual_differentiation == pytest.approx(
        once.perceptual_differentiation
    )


def test_disjoint_structures_sum_exactly(perceptions):
    # grid3's two stimuli trigger different states, so the structures share
    # no components and the union sums both D and D_p exactly.
    p0, p1 = perceptions[(0,)], perceptions[(1,)]
    d = Differentiation((p0, p1))
    assert d.differentiation == pytest.approx(
        float(p0.ces.big_phi) + float(p1.ces.big_phi)
    )
    assert d.perceptual_differentiation == pytest.approx(p0.richness + p1.richness)


def test_perceptual_differentiation_at_most_sum_of_richness(perceptions):
    d = Differentiation(tuple(perceptions.values()))
    total = sum(p.richness for p in perceptions.values())
    assert d.perceptual_differentiation <= total + 1e-12


# --- Analytical differentiation (closed-form D) -----------------------------


def test_analytical_matches_concrete_disjoint(perceptions):
    # grid3's two structures are disjoint: the |T|>=2 cross terms vanish, so
    # analytical D = sum of each structure's big_phi, same as concrete.
    d = Differentiation(tuple(perceptions.values()))
    assert d.analytical_differentiation == pytest.approx(d.differentiation)


def test_analytical_single_structure_is_big_phi(perceptions):
    p = perceptions[(0,)]
    d = Differentiation((p,))
    assert d.analytical_differentiation == pytest.approx(float(p.ces.big_phi))


def test_analytical_duplicate_structure_is_idempotent(perceptions):
    p = perceptions[(0,)]
    once = Differentiation((p,))
    twice = Differentiation((p, p))
    assert twice.analytical_differentiation == pytest.approx(
        once.analytical_differentiation
    )


def test_analytical_empty_is_zero():
    assert Differentiation(()).analytical_differentiation == 0.0


def _concrete_ces_subset(ces, n_keep):
    """A CES over the first ``n_keep`` distinctions of ``ces``, concrete relations."""
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.relations import relations as compute_relations

    sub_distinctions = ResolvedDistinctions(list(ces.distinctions)[:n_keep])
    sub_relations = compute_relations(sub_distinctions, "CONCRETE")
    return CauseEffectStructure(
        sia=ces.sia, distinctions=sub_distinctions, relations=sub_relations
    )


def test_analytical_partial_overlap_matches_concrete(perceptions):
    # A structure and a strict sub-structure of it: the union is the full
    # structure, so D = big_phi(full). The |T|=2 term restricts to the shared
    # (sub) distinctions, exercising the common-distinction restriction + sign.
    full = perceptions[(0,)]
    sub_ces = _concrete_ces_subset(full.ces, len(full.ces.distinctions) - 1)
    sub = Perception(ces=sub_ces, triggered_tpm=full.triggered_tpm, stimulus=(0,))
    d = Differentiation((full, sub))
    assert d.analytical_differentiation == pytest.approx(d.differentiation)
    assert d.analytical_differentiation == pytest.approx(float(full.ces.big_phi))


def test_analytical_runs_on_analytical_relations_where_concrete_cannot(perceptions):
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.relations import AnalyticalRelations

    concrete = perceptions[(0,)]
    analytical_ces = CauseEffectStructure(
        sia=concrete.ces.sia,
        distinctions=concrete.ces.distinctions,
        relations=AnalyticalRelations(concrete.ces.distinctions),
    )
    analytical_p = Perception(
        ces=analytical_ces,
        triggered_tpm=concrete.triggered_tpm,
        stimulus=(0,),
    )
    d_analytical = Differentiation((analytical_p,))
    d_concrete = Differentiation((concrete,))
    # The analytical path runs and agrees with the concrete value...
    assert d_analytical.analytical_differentiation == pytest.approx(
        d_concrete.differentiation
    )
    # ...while the concrete path cannot walk non-iterable AnalyticalRelations.
    with pytest.raises(TypeError):
        _ = d_analytical.differentiation
