"""Parity and invariant tests for the relations query surface.

The iterating backend (``ConcreteRelations``) and the closed-form backend
(``AnalyticalRelations``) must answer every query identically on systems
small enough to enumerate.
"""

import math

import numpy as np
import pytest

from pyphi import config
from pyphi import examples
from pyphi import numerics
from pyphi import relations
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure

NETWORKS = ["xor", "basic", "rule110", "fig4", "grid3"]


@pytest.fixture(scope="module", params=NETWORKS)
def structures(request):
    name = request.param
    with config.override(parallel=False):
        system = getattr(examples, f"{name}_system")()
        distinctions = new_big_phi.ces(
            system,
            system_measure=resolve_system_measure(
                config.formalism.iit.system_phi_measure
            ),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        ).distinctions
        concrete = relations.ConcreteRelations(relations.all_relations(distinctions))
    analytical = relations.AnalyticalRelations(distinctions)
    return name, distinctions, concrete, analytical


# --- Base (iterating) implementations, exercised via ConcreteRelations ---


def test_concrete_sum_phi_moment_first_moment_is_sum_phi(structures):
    _, _, concrete, _ = structures
    assert concrete.sum_phi_moment(1) == pytest.approx(concrete.sum_phi())


def test_concrete_phi_mean_std_matches_manual(structures):
    _, _, concrete, _ = structures
    if concrete.num_relations() == 0:
        with pytest.raises(ValueError):
            concrete.phi_mean_std()
        return
    phis = [float(r.phi) for r in concrete]
    mean, std = concrete.phi_mean_std()
    assert mean == pytest.approx(sum(phis) / len(phis))
    assert std == pytest.approx(
        math.sqrt(sum(p**2 for p in phis) / len(phis) - (sum(phis) / len(phis)) ** 2)
    )


def test_concrete_degree_spectrum_totals(structures):
    _, _, concrete, _ = structures
    spectrum = concrete.degree_spectrum()
    assert sum(count for count, _ in spectrum.values()) == concrete.num_relations()
    assert math.fsum(s for _, s in spectrum.values()) == pytest.approx(
        concrete.sum_phi()
    )
    assert all(count > 0 for count, _ in spectrum.values())


def test_concrete_degree_queries_match_iteration(structures):
    _, _, concrete, _ = structures
    for degree in range(1, max((len(r) for r in concrete), default=0) + 2):
        expected_count = sum(1 for r in concrete if len(r) == degree)
        expected_sum = math.fsum(float(r.phi) for r in concrete if len(r) == degree)
        assert concrete.num_relations_of_degree(degree) == expected_count
        assert concrete.sum_phi_of_degree(degree) == pytest.approx(expected_sum)


def test_concrete_max_phi(structures):
    _, _, concrete, _ = structures
    if concrete.num_relations() == 0:
        assert concrete.max_phi() == 0.0
        return
    assert concrete.max_phi() == pytest.approx(max(float(r.phi) for r in concrete))


def test_concrete_phi_histogram_totals(structures):
    _, _, concrete, _ = structures
    hist = concrete.phi_histogram()
    assert sum(hist.values()) == concrete.num_relations()
    assert math.fsum(phi * count for phi, count in hist.items()) == pytest.approx(
        concrete.sum_phi()
    )


def test_concrete_num_faces_matches_iteration(structures):
    _, _, concrete, _ = structures
    assert concrete.num_faces() == sum(r.num_faces for r in concrete)


def test_concrete_strongest_is_descending_and_complete(structures):
    _, _, concrete, _ = structures
    stream = list(concrete.strongest())
    phis = [float(r.phi) for r in stream]
    assert phis == sorted(phis, reverse=True)
    assert set(stream) == set(concrete)


def test_concrete_strongest_options(structures):
    _, _, concrete, _ = structures
    if concrete.num_relations() == 0:
        assert list(concrete.strongest(k=3)) == []
        return
    top3 = list(concrete.strongest(k=3))
    assert len(top3) == min(3, concrete.num_relations())
    pairs_only = list(concrete.strongest(max_degree=2))
    assert all(len(r) <= 2 for r in pairs_only)
    threshold = float(top3[-1].phi)
    above = list(concrete.strongest(min_phi=threshold))
    assert all(
        float(r.phi) > threshold or numerics.eq(float(r.phi), threshold) for r in above
    )


def test_concrete_materialize_filters(structures):
    _, _, concrete, _ = structures
    assert concrete.materialize() == concrete
    capped = concrete.materialize(max_degree=2)
    assert capped == relations.ConcreteRelations(r for r in concrete if len(r) <= 2)


def test_base_sample_not_implemented(structures):
    _, _, concrete, _ = structures
    with pytest.raises(NotImplementedError):
        concrete.sample(10, seed=0)


def test_null_relations_query_defaults():
    nr = relations.NullRelations()
    assert nr.sum_phi_moment(2) == 0.0
    assert nr.degree_spectrum() == {}
    assert nr.num_relations_of_degree(2) == 0
    assert nr.sum_phi_of_degree(2) == 0.0
    assert nr.max_phi() == 0.0
    assert nr.phi_histogram() == {}
    assert nr.binding_matrix().empty
    assert nr.num_faces() == 0
    assert list(nr.strongest()) == []
    assert nr.materialize() == relations.ConcreteRelations(())
    with pytest.raises(ValueError):
        nr.phi_mean_std()


# --- Analytical closed forms: parity with concrete enumeration ---


@pytest.mark.parametrize("k", [1, 2, 3])
def test_analytical_moments_match_concrete(structures, k):
    _, _, concrete, analytical = structures
    assert analytical.sum_phi_moment(k) == pytest.approx(concrete.sum_phi_moment(k))


def test_analytical_phi_mean_std_matches_concrete(structures):
    _, _, concrete, analytical = structures
    if concrete.num_relations() == 0:
        with pytest.raises(ValueError):
            analytical.phi_mean_std()
        return
    assert analytical.phi_mean_std() == pytest.approx(concrete.phi_mean_std())


def test_analytical_degree_queries_match_concrete(structures):
    _, _, concrete, analytical = structures
    for degree in range(1, max((len(r) for r in concrete), default=0) + 2):
        assert analytical.num_relations_of_degree(
            degree
        ) == concrete.num_relations_of_degree(degree)
        assert analytical.sum_phi_of_degree(degree) == pytest.approx(
            concrete.sum_phi_of_degree(degree)
        )


def test_analytical_degree_spectrum_matches_concrete(structures):
    _, _, concrete, analytical = structures
    analytical_spectrum = analytical.degree_spectrum()
    concrete_spectrum = concrete.degree_spectrum()
    assert analytical_spectrum.keys() == concrete_spectrum.keys()
    for degree in concrete_spectrum:
        assert analytical_spectrum[degree][0] == concrete_spectrum[degree][0]
        assert analytical_spectrum[degree][1] == pytest.approx(
            concrete_spectrum[degree][1]
        )


def test_analytical_max_phi_matches_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.max_phi() == pytest.approx(concrete.max_phi())


def _assert_histograms_match(left, right):
    """Histograms match if their sorted (key, count) sequences align with
    approx-equal keys and equal counts (keys are precision-rounded floats,
    so the two backends may differ by one unit in the last rounded place)."""
    left_items = sorted(left.items())
    right_items = sorted(right.items())
    assert len(left_items) == len(right_items)
    for (left_phi, left_count), (right_phi, right_count) in zip(
        left_items, right_items, strict=True
    ):
        assert left_phi == pytest.approx(right_phi, abs=1e-12)
        assert left_count == right_count


def test_analytical_phi_histogram_matches_concrete(structures):
    _, _, concrete, analytical = structures
    _assert_histograms_match(analytical.phi_histogram(), concrete.phi_histogram())


def test_analytical_phi_histogram_totals(structures):
    _, _, _, analytical = structures
    hist = analytical.phi_histogram()
    assert sum(hist.values()) == analytical.num_relations()
    assert math.fsum(phi * count for phi, count in hist.items()) == pytest.approx(
        analytical.sum_phi()
    )


def test_binding_matrix_parity(structures):
    _, _, concrete, analytical = structures
    concrete_matrix = concrete.binding_matrix()
    analytical_matrix = analytical.binding_matrix()
    assert list(concrete_matrix.index) == list(analytical_matrix.index)
    assert np.allclose(
        concrete_matrix.to_numpy(), analytical_matrix.to_numpy(), atol=1e-10
    )


def test_binding_matrix_is_symmetric_with_positive_diagonal(structures):
    _, _, _, analytical = structures
    matrix = analytical.binding_matrix()
    values = matrix.to_numpy()
    assert np.allclose(values, values.T, atol=1e-12)
    assert (np.diag(values) > 0).all()


def test_analytical_num_faces_matches_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.num_faces() == concrete.num_faces()


def test_analytical_strongest_matches_sorted_concrete(structures):
    _, _, concrete, analytical = structures
    analytical_stream = list(analytical.strongest())
    concrete_sorted = sorted(concrete, key=lambda r: float(r.phi), reverse=True)
    assert [float(r.phi) for r in analytical_stream] == pytest.approx(
        [float(r.phi) for r in concrete_sorted]
    )
    assert set(analytical_stream) == set(concrete)


def test_analytical_strongest_top_k(structures):
    _, _, concrete, analytical = structures
    k = 5
    top = list(analytical.strongest(k=k))
    assert len(top) == min(k, concrete.num_relations())
    concrete_top_phis = sorted((float(r.phi) for r in concrete), reverse=True)[
        : len(top)
    ]
    assert [float(r.phi) for r in top] == pytest.approx(concrete_top_phis)


def test_analytical_strongest_min_phi_and_max_degree(structures):
    _, _, concrete, analytical = structures
    if concrete.num_relations() == 0:
        assert list(analytical.strongest()) == []
        return
    phis = sorted((float(r.phi) for r in concrete), reverse=True)
    threshold = phis[len(phis) // 2]
    above = list(analytical.strongest(min_phi=threshold))
    expected = [p for p in phis if p > threshold or numerics.eq(p, threshold)]
    assert [float(r.phi) for r in above] == pytest.approx(expected)
    pairs = list(analytical.strongest(max_degree=2))
    assert all(len(r) <= 2 for r in pairs)
    expected_pairs = sorted(
        (float(r.phi) for r in concrete if len(r) <= 2), reverse=True
    )
    assert [float(r.phi) for r in pairs] == pytest.approx(expected_pairs)


# --- Sampling ---


def test_sample_is_seed_reproducible(structures):
    _, _, _, analytical = structures
    first = analytical.sample(200, seed=42)
    second = analytical.sample(200, seed=42)
    assert first.sum_phi() == second.sum_phi()
    assert first.num_relations() == second.num_relations()
    assert analytical.sample(200, seed=7).relations != first.relations or (
        analytical.num_relations() <= 1
    )


def test_sample_estimates_are_accurate(structures):
    _, _, concrete, analytical = structures
    sample = analytical.sample(2000, seed=42)
    exact_count = concrete.num_relations()
    exact_sum = concrete.sum_phi()
    count_estimate, count_stderr = sample.num_relations()
    sum_estimate, sum_stderr = sample.sum_phi()
    # Deterministic given the seed; generous but meaningful bounds.
    assert abs(count_estimate - exact_count) <= max(5 * count_stderr, 0.05 * exact_count)
    assert abs(sum_estimate - exact_sum) <= max(5 * sum_stderr, 0.05 * exact_sum)


def test_sample_estimate_of_predicate(structures):
    _, _, concrete, analytical = structures
    sample = analytical.sample(2000, seed=42)
    exact = sum(1 for r in concrete if not r.is_self_relation and len(r) == 2)
    estimate, stderr = sample.estimate(lambda r: 1.0 if len(r) == 2 else 0.0)
    assert abs(estimate - exact) <= max(5 * stderr, 0.05 * exact + 1.0)


def test_sample_metadata(structures):
    _, _, _, analytical = structures
    sample = analytical.sample(50, seed=3)
    assert sample.seed == 3
    # A structure with no non-self relations has normalization 0 and draws
    # nothing.
    assert len(sample) == (50 if sample.normalization > 0 else 0)
    assert all(len(r) >= 2 for r in sample)
    assert isinstance(sample.normalization, int)


def test_sample_requires_seed_keyword(structures):
    _, _, _, analytical = structures
    with pytest.raises(TypeError):
        analytical.sample(10, 42)  # seed must be keyword-only


def test_analytical_materialize_equals_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.materialize() == concrete


def test_analytical_materialize_bounds(structures):
    _, _, concrete, analytical = structures
    capped = analytical.materialize(max_degree=2)
    assert capped == relations.ConcreteRelations(r for r in concrete if len(r) <= 2)
    threshold = concrete.max_phi()
    top = analytical.materialize(min_phi=threshold)
    assert all(
        float(r.phi) > threshold or numerics.eq(float(r.phi), threshold) for r in top
    )
    assert len(top) >= min(1, concrete.num_relations())


def test_degree_guard_parity(structures):
    _, _, concrete, analytical = structures
    for degree in (0, -1):
        assert (
            analytical.num_relations_of_degree(degree)
            == concrete.num_relations_of_degree(degree)
            == 0
        )
        assert (
            analytical.sum_phi_of_degree(degree)
            == concrete.sum_phi_of_degree(degree)
            == 0.0
        )


def test_sum_phi_moment_zero_raises(structures):
    _, _, concrete, analytical = structures
    for backend in (concrete, analytical):
        with pytest.raises(ValueError):
            backend.sum_phi_moment(0)


def test_strongest_k_zero_yields_nothing(structures):
    _, _, concrete, analytical = structures
    for backend in (concrete, analytical):
        assert list(backend.strongest(k=0)) == []


# --- Folds: every query restricted to incident relations ---


@pytest.fixture(scope="module", params=[1, 2])
def fold_structures(structures, request):
    name, distinctions, concrete, _ = structures
    all_distinctions = list(distinctions)
    seeds = all_distinctions[: min(request.param, len(all_distinctions))]
    seed_set = set(seeds)
    fold = relations.AnalyticalFoldRelations(distinctions, seeds)
    incident_concrete = relations.ConcreteRelations(
        r for r in concrete if not seed_set.isdisjoint(r)
    )
    return name, seeds, fold, incident_concrete


def test_fold_moments_match_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    for k in (1, 2):
        assert fold.sum_phi_moment(k) == pytest.approx(incident.sum_phi_moment(k))


def test_fold_phi_mean_std_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    if incident.num_relations() == 0:
        with pytest.raises(ValueError):
            fold.phi_mean_std()
        return
    assert fold.phi_mean_std() == pytest.approx(incident.phi_mean_std())


def test_fold_degree_spectrum_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    fold_spectrum = fold.degree_spectrum()
    incident_spectrum = incident.degree_spectrum()
    assert fold_spectrum.keys() == incident_spectrum.keys()
    for degree in incident_spectrum:
        assert fold_spectrum[degree][0] == incident_spectrum[degree][0]
        assert fold_spectrum[degree][1] == pytest.approx(incident_spectrum[degree][1])


def test_fold_max_phi_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.max_phi() == pytest.approx(incident.max_phi())


def test_fold_phi_histogram_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    _assert_histograms_match(fold.phi_histogram(), incident.phi_histogram())


def test_fold_num_faces_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.num_faces() == incident.num_faces()


def test_fold_binding_matrix_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    fold_matrix = fold.binding_matrix()
    incident_matrix = incident.binding_matrix()
    assert set(incident_matrix.index) <= set(fold_matrix.index)
    aligned = incident_matrix.reindex(
        index=fold_matrix.index, columns=fold_matrix.columns, fill_value=0.0
    )
    assert np.allclose(fold_matrix.to_numpy(), aligned.to_numpy(), atol=1e-10)


def test_fold_strongest_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    stream = list(fold.strongest())
    assert [float(r.phi) for r in stream] == pytest.approx(
        sorted((float(r.phi) for r in incident), reverse=True)
    )
    assert set(stream) == set(incident)


def test_fold_materialize_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.materialize() == incident


def test_fold_sample_not_implemented(fold_structures):
    _, _, fold, _ = fold_structures
    with pytest.raises(NotImplementedError):
        fold.sample(10, seed=0)


def test_sum_phi_by_distinction_parity(structures):
    """Per-distinction incident Σφ_r agrees between the iterating and
    closed-form backends, and the iterating sum conserves Σ_r φ_r·degree(r)."""
    _, distinctions, concrete, analytical = structures
    dl = list(distinctions)
    conc = concrete.sum_phi_by_distinction(dl)
    anal = analytical.sum_phi_by_distinction(dl)
    assert len(conc) == len(dl)
    for c, a in zip(conc, anal, strict=True):
        assert c == pytest.approx(a)
    # Independent oracle: each relation contributes φ_r to each of its relata.
    assert sum(conc) == pytest.approx(sum(float(r.phi) * len(r) for r in concrete))


def test_fold_sum_phi_by_distinction_matches_incident_concrete(
    fold_structures, structures
):
    """Per-distinction incident Σφ_r on a fold equals iterating the fold's
    concrete incident relations."""
    _, _, fold, incident = fold_structures
    _, distinctions, _, _ = structures
    dl = list(distinctions)
    expected = incident.sum_phi_by_distinction(dl)
    actual = fold.sum_phi_by_distinction(dl)
    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e)


# --- Maximal relations and maximal faces (facets of the relation complex) ---


class _StubMICE:
    """Minimal stand-in for a cause/effect side: a parent and purview units."""

    def __init__(self, label, parent, purview_units):
        self._label = label
        self.parent = parent
        self.purview_units = frozenset(purview_units)

    def __repr__(self):
        return self._label


class _StubDistinction:
    """Minimal stand-in for a distinction, enough for Relation/RelationFace."""

    def __init__(self, mechanism, phi, cause_units, effect_units):
        self.mechanism = mechanism
        self.phi = phi
        self.cause = _StubMICE(f"{mechanism}.cause", self, cause_units)
        self.effect = _StubMICE(f"{mechanism}.effect", self, effect_units)

    @property
    def purview_union(self):
        return set(self.cause.purview_units | self.effect.purview_units)

    def __repr__(self):
        return f"D{self.mechanism}"


@pytest.fixture
def stub_distinctions():
    """Three overlapping distinctions plus one isolated one.

    Z(x) = {A, B, C}, Z(y) = {A, B}, Z(z) = {E} (isolated).
    The maximal face at atom y has parents {A, B} — a NON-maximal relation —
    because A's cause side (the only side of A containing y) does not
    contain x, so M(y) is not a subset of M(x).
    """
    a = _StubDistinction((0,), 1.0, cause_units={"y"}, effect_units={"x"})
    b = _StubDistinction((1,), 1.0, cause_units={"x", "y"}, effect_units={"x"})
    c = _StubDistinction((2,), 1.0, cause_units={"x"}, effect_units={"x"})
    e = _StubDistinction((3,), 1.0, cause_units={"z"}, effect_units={"z"})
    return a, b, c, e


def test_maximal_relations_stub(stub_distinctions):
    a, b, c, e = stub_distinctions
    result = relations.maximal_relations([a, b, c, e])
    assert isinstance(result, relations.ConcreteRelations)
    assert {frozenset(r) for r in result} == {frozenset({a, b, c})}
    (facet,) = result
    # φ_r(ABC) = |{x}| · min(1/2, 1/2, 1)
    assert float(facet.phi) == pytest.approx(0.5)


def test_maximal_faces_stub_parent_relation_may_be_non_maximal(stub_distinctions):
    a, b, c, e = stub_distinctions
    faces = relations.maximal_faces([a, b, c, e])
    m_x = frozenset({a.effect, b.cause, b.effect, c.cause, c.effect})
    m_y = frozenset({a.cause, b.cause})
    assert {frozenset(f) for f in faces} == {m_x, m_y}
    by_content = {frozenset(f): f for f in faces}
    # M(x) carries φ_r(Z(x)) = φ_r({A,B,C}); M(y) carries φ_r(Z(y)) = φ_r({A,B}).
    assert float(by_content[m_x].phi) == pytest.approx(0.5)
    assert float(by_content[m_y].phi) == pytest.approx(1.0)
    # The key property: M(y)'s parents {A, B} are NOT a maximal relation.
    maximal_relata = {frozenset(r) for r in relations.maximal_relations([a, b, c, e])}
    assert frozenset({a, b}) not in maximal_relata


def test_maximal_functions_empty_and_isolated(stub_distinctions):
    *_, e = stub_distinctions
    assert set(relations.maximal_relations([])) == set()
    assert relations.maximal_faces([]) == frozenset()
    # A lone distinction relates to nothing: no degree-2 group exists.
    assert set(relations.maximal_relations([e])) == set()
    assert relations.maximal_faces([e]) == frozenset()


def test_maximal_relations_atom_filter(stub_distinctions):
    a, b, c, e = stub_distinctions
    # Restricting to atom y keeps only Z(y) = {A, B}.
    result = relations.maximal_relations([a, b, c, e], atoms={"y"})
    assert {frozenset(r) for r in result} == {frozenset({a, b})}
