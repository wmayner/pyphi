"""Tests for the 3-D simplicial-complex renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def test_geometry_dataclass_frozen():
    import dataclasses

    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    geo = SimplicialComplexGeometry()
    assert geo.max_radius == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        geo.max_radius = 2.0  # type: ignore[misc]


def test_endpoint_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _endpoint_positions

    geo = SimplicialComplexGeometry()
    pos = _endpoint_positions(xor_projection, geo)
    assert set(pos) == set(range(8))
    # Deterministic.
    assert pos == _endpoint_positions(xor_projection, geo)
    # By default, z stacks by purview size: size-1 purviews below size-3.
    assert pos[1][2] < pos[0][2]
    # Flat when requested.
    flat = _endpoint_positions(xor_projection, SimplicialComplexGeometry(z_spacing=0.0))
    assert all(p[2] == 0.0 for p in flat.values())
    # d3's cause/effect share the purview (0,1,2): cause sits -x, effect +x.
    assert pos[6][0] < pos[7][0]
    # Endpoints sharing (purview, direction) are jittered apart:
    # causes of d0, d1, d2, d3 all have purview (0,1,2).
    cause_ids = (0, 2, 4, 6)
    assert len({pos[i] for i in cause_ids}) == 4


def test_mechanism_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _mechanism_positions

    geo = SimplicialComplexGeometry(max_radius=2.0)
    pos = _mechanism_positions(xor_projection, geo)
    assert set(pos) == {0, 1, 2, 3}
    # Mechanisms are unique, so all positions distinct.
    assert len(set(pos.values())) == 4
    # abc is the only size-3 mechanism: a single-member shell sits on the
    # central axis rather than off on its polygon.
    x, y, _z = pos[3]
    assert (x, y) == (0.0, 0.0)
    # The three pairs share the size-2 shell at its radius.
    for i in (0, 1, 2):
        px, py, _pz = pos[i]
        assert (px**2 + py**2) ** 0.5 == pytest.approx(2.0 * 2 / 3)


@pytest.fixture
def antipodal_projection():
    """Four singleton purviews whose faces pair them antipodally under
    lexicographic ring placement: (0,)-(2,) and (1,)-(3,). An angular
    barycentric layout makes each pair adjacent, shortening the faces.
    """
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import DistinctionNode
    from pyphi.visualize.projection import EndpointNode
    from pyphi.visualize.projection import InclusionOrder
    from pyphi.visualize.projection import PhiStructureProjection
    from pyphi.visualize.projection import RelationFaceEdge

    def node(i):
        return DistinctionNode(
            id=i,
            mechanism=(i,),
            label="abcd"[i],
            cause_purview=(i,),
            effect_purview=(i,),
            mechanism_state=(0,),
            phi=1.0,
            sum_phi_relations=0.0,
            includes=False,
            included=False,
        )

    def endpoint(eid, i, direction):
        return EndpointNode(
            id=eid,
            distinction_id=i,
            direction=direction,
            purview=(i,),
            purview_state=(0,),
            phi=1.0,
            label="abcd"[i],
        )

    endpoints = tuple(
        endpoint(2 * i + j, i, d)
        for i in range(4)
        for j, d in enumerate(("cause", "effect"))
    )
    faces = (
        RelationFaceEdge(endpoints=(0, 4), degree=2, phi=1.0, overlap=()),
        RelationFaceEdge(endpoints=(2, 6), degree=2, phi=1.0, overlap=()),
    )
    order = InclusionOrder(covers=((), (), (), ()), rank=(0, 0, 0, 0), size=(1, 1, 1, 1))
    return PhiStructureProjection(
        nodes=tuple(node(i) for i in range(4)),
        edges=(),
        mechanism_inclusion=order,
        purview_union_inclusion=order,
        node_labels=NodeLabels(("A", "B", "C", "D"), (0, 1, 2, 3)),
        endpoints=endpoints,
        faces=faces,
    )


def _total_face_length(projection, pos):
    from itertools import combinations

    total = 0.0
    for f in projection.faces:
        for a, b in combinations((pos[i] for i in f.endpoints), 2):
            total += sum((u - v) ** 2 for u, v in zip(a, b, strict=True)) ** 0.5
    return total


def test_barycentric_layout_shortens_faces(antipodal_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _positions_3d

    geo = SimplicialComplexGeometry()
    sorted_pos, _ = _positions_3d(antipodal_projection, geo, layout="sorted")
    bary_pos, _ = _positions_3d(antipodal_projection, geo, layout="barycentric")
    assert _total_face_length(antipodal_projection, bary_pos) < _total_face_length(
        antipodal_projection, sorted_pos
    )
    # Deterministic.
    again, _ = _positions_3d(antipodal_projection, geo, layout="barycentric")
    assert again == bary_pos


def test_unknown_3d_layout_raises(antipodal_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _positions_3d

    with pytest.raises(ValueError, match="layout"):
        _positions_3d(antipodal_projection, SimplicialComplexGeometry(), layout="bogus")


def _render(projection, **kwargs):
    from pyphi.visualize.render.simplicial_complex import render_simplicial_complex
    from pyphi.visualize.theme import DEFAULT_THEME

    return render_simplicial_complex(projection, DEFAULT_THEME, **kwargs)


def test_render_full_figure_structure(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection)
    assert isinstance(fig, go.Figure)
    # One trace per element class, in declaration order.
    assert len(fig.data) == 6
    purviews, mechanisms, ce_links, mp_links, two_faces, mesh = fig.data
    assert len(purviews.x) == 8
    assert len(mechanisms.x) == 4
    # Cause-effect links: (cause, effect, None) per distinction.
    assert len(ce_links.x) == 3 * 4
    # Mechanism-purview links: (cause, mechanism, effect, None) per distinction.
    assert len(mp_links.x) == 4 * 4
    # 25 degree-2 faces, (a, b, None) each.
    assert len(two_faces.x) == 3 * 25
    # 40 degree-3 faces as one mesh.
    assert isinstance(mesh, go.Mesh3d)
    assert len(mesh.i) == 40
    # Endpoint labels present.
    assert "abc" in purviews.text and "c" in purviews.text


def test_render_show_subsetting(xor_projection):
    fig = _render(xor_projection, show=("purviews",))
    assert len(fig.data) == 1
    fig = _render(xor_projection, show=("purviews", "three_faces"))
    assert len(fig.data) == 2
    with pytest.raises(ValueError, match="show"):
        _render(xor_projection, show=("purviews", "bogus"))


def test_render_only_distinctions_filters_without_moving(xor_projection):
    full = _render(xor_projection)
    sub = _render(xor_projection, only_distinctions={0, 3})
    full_points = set(zip(full.data[0].x, full.data[0].y, full.data[0].z, strict=True))
    sub_points = set(zip(sub.data[0].x, sub.data[0].y, sub.data[0].z, strict=True))
    # 2 distinctions -> 4 endpoints, at unchanged coordinates.
    assert len(sub_points) == 4
    assert sub_points <= full_points
    # Faces restricted to those entirely within the subset.
    import plotly.graph_objects as go

    full_mesh = next(t for t in full.data if isinstance(t, go.Mesh3d))
    sub_meshes = [t for t in sub.data if isinstance(t, go.Mesh3d)]
    if sub_meshes:
        assert len(sub_meshes[0].i) < len(full_mesh.i)


def test_plot_phi_structure_simplicial_complex_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    ces = examples.xor_system().ces()
    fig = plot_phi_structure(ces, view="simplicial_complex")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6
    fig = plot_phi_structure(
        ces,
        view="simplicial_complex",
        geometry=SimplicialComplexGeometry(z_spacing=0.3),
        show=("purviews",),
    )
    assert len(fig.data) == 1
    # The shared layout knob applies to this view too.
    for layout in ("barycentric", "sorted"):
        fig = plot_phi_structure(ces, view="simplicial_complex", layout=layout)
        assert len(fig.data) == 6


def test_highlight_phi_fold_smoke():
    from types import SimpleNamespace

    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = SimpleNamespace(distinctions=list(ces.distinctions)[:2])
    fig = highlight_phi_fold(ces, fold)
    # Two passes: dimmed full structure + highlighted fold.
    assert len(fig.data) == 12
    # The overlay's endpoint coordinates are a subset of the background's.
    bg, overlay = fig.data[0], fig.data[6]
    bg_points = set(zip(bg.x, bg.y, bg.z, strict=True))
    overlay_points = set(zip(overlay.x, overlay.y, overlay.z, strict=True))
    assert len(overlay.x) == 4
    assert overlay_points <= bg_points
