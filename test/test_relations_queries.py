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
    assert nr.max_phi() == 0.0
    assert nr.phi_histogram() == {}
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
