"""Tests for the pure visualization projection layer."""

import pytest

from pyphi.visualize.projection import InclusionOrder
from pyphi.visualize.projection import _inclusion_order


def test_inclusion_order_exact():
    # purview-unit sets: two points, a pair, the whole.
    unions = (
        frozenset({0}),
        frozenset({1}),
        frozenset({0, 1}),
        frozenset({0, 1, 2}),
    )
    order = _inclusion_order(unions)
    assert isinstance(order, InclusionOrder)
    # covers = transitive reduction: the whole covers only the pair;
    # the pair covers both points; points cover nothing.
    assert order.covers == ((), (), (0, 1), (2,))
    # rank = longest down-chain: points 0, pair 1, whole 2.
    assert order.rank == (0, 0, 1, 2)
    # size = cardinality of each set.
    assert order.size == (1, 1, 2, 3)


def test_inclusion_order_equal_unions_no_edge():
    # Mutually-equal purview unions: same rank, no cover edge between them.
    unions = (frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    order = _inclusion_order(unions)
    assert order.covers == ((2,), (2,), ())
    assert order.rank == (1, 1, 0)


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    ces = examples.xor_system().ces()
    return project_ces(ces)


def test_project_xor_nodes(xor_projection):
    proj = xor_projection
    assert len(proj.nodes) == 4
    assert [n.mechanism for n in proj.nodes] == [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    assert [n.label for n in proj.nodes] == ["ab", "ac", "bc", "abc"]
    # First distinction's observed values (xor in (0,0,0)):
    assert proj.nodes[0].cause_purview == (0, 1, 2)
    assert proj.nodes[0].effect_purview == (2,)
    assert proj.nodes[0].phi == pytest.approx(0.5)


def test_project_xor_edges(xor_projection):
    proj = xor_projection
    assert len(proj.edges) == 15
    for e in proj.edges:
        assert all(0 <= i < 4 for i in e.relata)
        # Relation is a frozenset of relata, so a self-relation whose cause
        # and effect relata coincide has degree 1.
        assert e.degree >= 1
        assert e.phi >= 0


def test_project_xor_sum_phi_relations_consistent(xor_projection):
    proj = xor_projection
    # Recompute independently from the projected edges.
    expected = [0.0] * 4
    for e in proj.edges:
        for i in e.relata:
            expected[i] += e.phi
    for node, exp in zip(proj.nodes, expected, strict=True):
        assert node.sum_phi_relations == pytest.approx(exp)


def test_project_xor_inclusion_monotone(xor_projection):
    proj = xor_projection
    for order in (proj.mechanism_inclusion, proj.purview_union_inclusion):
        # rank strictly decreases along covers edges.
        for a, cov in enumerate(order.covers):
            for b in cov:
                assert order.rank[a] > order.rank[b]


def test_project_xor_mechanism_inclusion_exact(xor_projection):
    # Mechanisms ab, ac, bc, abc: the pairs are incomparable; abc covers all.
    order = xor_projection.mechanism_inclusion
    assert order.covers == ((), (), (), (0, 1, 2))
    assert order.rank == (0, 0, 0, 1)
    assert order.size == (2, 2, 2, 3)


def test_projection_inclusion_accessor(xor_projection):
    proj = xor_projection
    assert proj.inclusion("mechanism") is proj.mechanism_inclusion
    assert proj.inclusion("purview_union") is proj.purview_union_inclusion
    with pytest.raises(ValueError, match="order"):
        proj.inclusion("bogus")


def test_state_cased_label():
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import _state_cased_label

    nl = NodeLabels(("A", "B", "C"), (0, 1, 2))
    assert _state_cased_label((0, 2), (1, 0), nl) == "Ac"
    assert _state_cased_label((0, 1, 2), (0, 0, 0), nl) == "abc"


def test_project_xor_endpoints_exact(xor_projection):
    eps = xor_projection.endpoints
    assert len(eps) == 8
    assert tuple(e.id for e in eps) == tuple(range(8))
    # Interleaved cause/effect per distinction.
    assert [e.direction for e in eps] == ["cause", "effect"] * 4
    assert [e.distinction_id for e in eps] == [0, 0, 1, 1, 2, 2, 3, 3]
    # d0 = ab: cause over the whole substrate, effect on c.
    assert eps[0].purview == (0, 1, 2)
    assert eps[0].purview_state == (0, 0, 0)
    assert eps[0].phi == pytest.approx(0.5)
    assert eps[0].label == "abc"
    assert eps[1].purview == (2,)
    assert eps[1].phi == pytest.approx(1.0)
    assert eps[1].label == "c"
    # d3 = abc: effect phi 2.
    assert eps[7].purview == (0, 1, 2)
    assert eps[7].phi == pytest.approx(2.0)


def test_project_xor_faces(xor_projection):
    faces = xor_projection.faces
    by_degree = {}
    for f in faces:
        by_degree.setdefault(f.degree, []).append(f)
        assert f.degree == len(f.endpoints)
        assert all(0 <= i < 8 for i in f.endpoints)
        assert f.phi >= 0
    # All face degrees are carried now (degree >= 4 used to be dropped).
    assert {d: len(v) for d, v in sorted(by_degree.items())} == {
        2: 25,
        3: 40,
        4: 35,
        5: 16,
        6: 3,
    }
    # Known face: cause and effect of d2 (bc) related over unit a.
    assert any(
        f.endpoints == (4, 5) and f.phi == pytest.approx(1 / 6) and f.overlap == (0,)
        for f in by_degree[2]
    )


def test_projection_faces_deterministic(xor_projection):
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    again = project_ces(examples.xor_system().ces())
    assert again.faces == xor_projection.faces
    assert again.endpoints == xor_projection.endpoints


def test_theme_role_colors_cover_none():
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert set(roles) == {"extended", "includes", "included", "none"}


def test_theme_hypergraph_fields():
    from pyphi.visualize.theme import DEFAULT_THEME

    assert DEFAULT_THEME.cause_color == "#8D3D00"
    assert DEFAULT_THEME.effect_color == "#006146"
    assert DEFAULT_THEME.face_colorscale == "Blues"
    assert 0 < DEFAULT_THEME.face_opacity <= 1
    assert DEFAULT_THEME.text_size > 0
    # Spokes are straight by default (curvature is an opt-in) and a flat
    # neutral grey.
    assert DEFAULT_THEME.spoke_curvature == 0.0
    assert DEFAULT_THEME.spoke_width > 0
    assert DEFAULT_THEME.spoke_color.startswith("rgba")
    # Relation faces fade by opacity on a fixed hue; hubs have their own size
    # range, smaller than the purview markers'.
    assert len(DEFAULT_THEME.relation_rgb) == 3
    assert DEFAULT_THEME.relation_alpha_range[0] < DEFAULT_THEME.relation_alpha_range[1]
    assert DEFAULT_THEME.hub_size_range[0] < DEFAULT_THEME.hub_size_range[1]


def test_hypergraph_geometry_defaults():
    from pyphi.visualize.render.hypergraph import HypergraphGeometry

    geo = HypergraphGeometry()
    # Coincident purviews are separated by leaning toward their mechanism by
    # default; the prior even-polygon spread stays reachable.
    assert geo.endpoint_placement == "mechanism_anchored"
    # The global-embedding layout defaults to the MDS method.
    assert geo.embedding_method == "mds"
    # Embedding block weights default so purview dominates mechanism and
    # direction; they are exposed as knobs.
    assert geo.embed_purview_weight == 1.0
    assert geo.embed_mechanism_weight == 0.5
    assert geo.embed_direction_weight == 0.5
    # The embedding is volumetric (3-D) by default; planar flattening is opt-in.
    assert geo.embed_planar is False


def test_theme_frozen_with_defaults():
    import dataclasses

    from pyphi.visualize.theme import DEFAULT_THEME
    from pyphi.visualize.theme import Theme

    assert isinstance(DEFAULT_THEME, Theme)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THEME.colorscale = "Plasma"  # type: ignore[misc]
    dark = dataclasses.replace(DEFAULT_THEME, background="black")
    assert dark.background == "black" and DEFAULT_THEME.background == "white"


def test_project_rejects_fold_accepts_induced():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    with pytest.raises(TypeError, match="relation-closed"):
        project_ces(fold)
    induced = ces.induce(list(ces.distinctions)[:3])
    projection = project_ces(induced)
    assert len(projection.nodes) == 3


def _xor_ces(analytical):
    import pyphi
    from pyphi import examples

    if not analytical:
        return examples.xor_system().ces()
    with pyphi.config.override(relation_computation="ANALYTICAL"):
        return examples.xor_system().ces()


def test_project_ces_carries_degree_spectrum():
    from pyphi.visualize.projection import project_ces

    ces = _xor_ces(analytical=False)
    proj = project_ces(ces)
    assert proj.degree_spectrum == ces.relations.degree_spectrum()


def test_project_ces_analytical_requires_cap():
    from pyphi.relations import AnalyticalRelations
    from pyphi.visualize.projection import project_ces

    ces = _xor_ces(analytical=True)
    assert isinstance(ces.relations, AnalyticalRelations)  # precondition
    with pytest.raises(ValueError, match="max_relations"):
        project_ces(ces)


def test_project_ces_analytical_matches_concrete_and_sizes_faithfully():
    from pyphi.visualize.projection import project_ces

    concrete = project_ces(_xor_ces(analytical=False))
    n = len(concrete.edges)
    analytical = project_ces(_xor_ces(analytical=True), max_relations=n)
    # Same rendered relations (as relata sets) when the cap covers all.
    assert {e.relata for e in analytical.edges} == {e.relata for e in concrete.edges}
    # Node sizing is faithful: identical to the concrete incident Σφ_r.
    assert [node.sum_phi_relations for node in analytical.nodes] == pytest.approx(
        [node.sum_phi_relations for node in concrete.nodes]
    )
    # And independent of the cap: a tight cap does not change node sizes.
    capped = project_ces(_xor_ces(analytical=True), max_relations=1)
    assert [node.sum_phi_relations for node in capped.nodes] == pytest.approx(
        [node.sum_phi_relations for node in analytical.nodes]
    )
    assert len(capped.edges) <= 1
