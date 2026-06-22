"""Tests for the 3-D simplicial-complex renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    return project_ces(examples.xor_system().ces())


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


def _fake_projection(distinction_ids):
    """A minimal stand-in exposing ``endpoints[eid].distinction_id``."""
    from types import SimpleNamespace

    n = max(distinction_ids) + 1 if distinction_ids else 0
    endpoints = [
        SimpleNamespace(distinction_id=None) for _ in range(len(distinction_ids))
    ]
    for eid, did in distinction_ids.items():
        endpoints[eid] = SimpleNamespace(distinction_id=did)
    return SimpleNamespace(endpoints=endpoints), n


def test_anchored_offsets_point_toward_mechanism():
    import math

    from pyphi.visualize.render.simplicial_complex import _anchored_offsets

    # Two endpoints sharing a purview, mechanisms in distinct directions from
    # the shell base: mech 0 at +x, mech 1 at +y.
    proj, _ = _fake_projection({0: 0, 1: 1})
    mech = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}
    offsets = _anchored_offsets([0, 1], (0.0, 0.0), mech, proj, 0.1)
    # Each leans toward its own mechanism, at radius == jitter.
    assert offsets[0] == pytest.approx((0.1, 0.0))
    assert offsets[1] == pytest.approx((0.0, 0.1))
    assert all(math.hypot(*o) == pytest.approx(0.1) for o in offsets.values())


def test_anchored_offsets_collinear_tie_break():
    import math

    from pyphi.visualize.render.simplicial_complex import _anchored_offsets

    # Both mechanisms lie in the same direction (+x) from the base: the
    # directional pass would collide, so the tie-break spreads them apart.
    proj, _ = _fake_projection({0: 0, 1: 1})
    mech = {0: (1.0, 0.0, 0.0), 1: (2.0, 0.0, 0.0)}
    offsets = _anchored_offsets([0, 1], (0.0, 0.0), mech, proj, 0.1)
    assert offsets[0] != offsets[1]  # collision-free
    assert all(math.hypot(*o) == pytest.approx(0.1) for o in offsets.values())


def test_anchored_offsets_degenerate_falls_back():
    import math

    from pyphi.visualize.render.simplicial_complex import _anchored_offsets

    # Mechanisms coincident with the base (zero direction) carry no angle and
    # fall to the polygon spread; still distinct, still at radius jitter.
    proj, _ = _fake_projection({0: 0, 1: 1})
    mech = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}
    offsets = _anchored_offsets([0, 1], (0.0, 0.0), mech, proj, 0.1)
    assert offsets[0] != offsets[1]
    assert all(math.hypot(*o) == pytest.approx(0.1) for o in offsets.values())


def test_anchored_endpoints_are_distinct_and_on_radius(xor_projection):
    import math

    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _endpoint_positions
    from pyphi.visualize.render.simplicial_complex import _mechanism_positions
    from pyphi.visualize.render.simplicial_complex import _shell_positions

    geo = SimplicialComplexGeometry()  # mechanism_anchored is the default
    mech = _mechanism_positions(xor_projection, geo)
    pos = _endpoint_positions(xor_projection, geo, mechanism_pos=mech)
    base = _shell_positions((e.purview for e in xor_projection.endpoints), geo)
    # The four causes sharing purview (0,1,2) stay distinct and each sits at
    # exactly purview_jitter from the (direction-shifted) shell point.
    for eid in (0, 2, 4, 6):
        bx, by, _bz = base[xor_projection.endpoints[eid].purview]
        bx -= geo.direction_offset  # cause shift
        assert math.hypot(pos[eid][0] - bx, pos[eid][1] - by) == pytest.approx(0.1)
    assert len({pos[i] for i in (0, 2, 4, 6)}) == 4


def test_polygon_placement_still_available(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _endpoint_positions

    anchored = _endpoint_positions(xor_projection, SimplicialComplexGeometry())
    polygon = _endpoint_positions(
        xor_projection, SimplicialComplexGeometry(endpoint_placement="polygon")
    )
    # Both keep all endpoints distinct where they share a purview, but place
    # them differently.
    assert len({polygon[i] for i in (0, 2, 4, 6)}) == 4
    assert anchored != polygon


def test_unknown_endpoint_placement_raises(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _endpoint_positions

    with pytest.raises(ValueError, match="endpoint_placement"):
        _endpoint_positions(
            xor_projection, SimplicialComplexGeometry(endpoint_placement="bogus")
        )


def test_embedding_layout_distinct_normalized_and_centroids(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _positions_3d

    geo = SimplicialComplexGeometry(max_radius=1.0)  # embedding_method defaults to pca
    epos, mpos = _positions_3d(xor_projection, geo, layout="embedding")
    # One point per endpoint, all distinct.
    assert set(epos) == set(range(8))
    assert len(set(epos.values())) == 8
    # Normalized so the bounding box fits max_radius (plus the small
    # coincident-spread tolerance).
    assert all(max(abs(x), abs(y), abs(z)) <= 1.0 + 0.05 for x, y, z in epos.values())
    # A mechanism sits at the centroid of its two endpoints.
    for d in range(4):
        cx, cy, cz = epos[2 * d]
        ex, ey, ez = epos[2 * d + 1]
        assert mpos[d] == pytest.approx(((cx + ex) / 2, (cy + ey) / 2, (cz + ez) / 2))


def test_embedding_layout_deterministic(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _positions_3d

    for method in ("pca", "mds"):
        geo = SimplicialComplexGeometry(embedding_method=method)
        a = _positions_3d(xor_projection, geo, layout="embedding")
        b = _positions_3d(xor_projection, geo, layout="embedding")
        assert a == b


def test_embedding_method_validation(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _positions_3d

    geo = SimplicialComplexGeometry(embedding_method="bogus")
    with pytest.raises(ValueError, match="embedding_method"):
        _positions_3d(xor_projection, geo, layout="embedding")


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
    from pyphi.visualize.projection import CESProjection
    from pyphi.visualize.projection import DistinctionNode
    from pyphi.visualize.projection import EndpointNode
    from pyphi.visualize.projection import InclusionOrder
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
    return CESProjection(
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

    # star_min_degree=4 exercises every geometric form: degree-2 lines,
    # degree-3 mesh, and the degree->=4 star expansion.
    fig = _render(xor_projection, star_min_degree=4)
    assert isinstance(fig, go.Figure)
    # One trace per base element class, plus two for the degree->=4 star
    # expansion (hub markers + spokes).
    assert len(fig.data) == 8
    purviews, mechanisms, ce_links, mp_links, two_faces, mesh, hub, spokes = fig.data
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
    # 54 degree->=4 faces as one hub trace + one spoke trace.
    assert not isinstance(hub, go.Mesh3d)
    assert hub.mode == "markers"
    assert spokes.mode == "lines"
    assert len(hub.x) == 54
    # Endpoint labels present.
    assert "abc" in purviews.text and "c" in purviews.text


class _Face:
    endpoints = (0, 1, 2, 3)
    degree = 4
    phi = 0.5
    overlap = (0,)


_SQUARE_POS = {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0), 3: (2, 2, 0)}


def test_higher_face_trace_hub_at_centroid():
    import dataclasses

    from pyphi.visualize.render.simplicial_complex import _higher_face_trace
    from pyphi.visualize.theme import DEFAULT_THEME

    # Straight spokes so the per-spoke coordinate count is exact.
    theme = dataclasses.replace(DEFAULT_THEME, spoke_curvature=0.0)
    hub_trace, spoke_trace = _higher_face_trace([_Face()], _SQUARE_POS, theme)
    # Hub sits at the centroid of the four endpoints.
    assert (hub_trace.x[0], hub_trace.y[0], hub_trace.z[0]) == (1.0, 1.0, 0.0)
    # One hub marker per face.
    assert len(hub_trace.x) == 1
    # Four straight spokes, each (hub, endpoint, None) -> 3 coords.
    assert len(spoke_trace.x) == 3 * 4


def test_higher_face_spokes_curve_when_curvature_set():
    import dataclasses

    from pyphi.visualize.render.simplicial_complex import _SPOKE_ARC_SEGMENTS
    from pyphi.visualize.render.simplicial_complex import _higher_face_trace
    from pyphi.visualize.theme import DEFAULT_THEME

    theme = dataclasses.replace(DEFAULT_THEME, spoke_curvature=0.3)
    _, spoke_trace = _higher_face_trace([_Face()], _SQUARE_POS, theme)
    # Each spoke is now a sampled arc: (segments + 1) points + a None gap.
    assert len(spoke_trace.x) == 4 * (_SPOKE_ARC_SEGMENTS + 1 + 1)
    # The arc bows off the straight hub->endpoint chord: the midpoint of the
    # first spoke is not the chord midpoint. Hub is (1, 1, 0), endpoint 0 is
    # (0, 0, 0); the straight midpoint would be (0.5, 0.5, 0.0).
    mid = _SPOKE_ARC_SEGMENTS // 2
    assert (spoke_trace.x[mid], spoke_trace.y[mid], spoke_trace.z[mid]) != (
        0.5,
        0.5,
        0.0,
    )


def test_face_hover_names_overlap_and_relata(xor_projection):
    from pyphi.visualize.render.simplicial_complex import _face_hover_fn

    hover = _face_hover_fn(xor_projection)
    # The known degree-2 face: cause/effect of bc related over unit a.
    face = next(
        f for f in xor_projection.faces if f.degree == 2 and f.endpoints == (4, 5)
    )
    text = hover(face)
    # Degree and phi.
    assert "2-face" in text and "φ =" in text
    # Overlap is labelled (unit a), not the raw index tuple.
    assert "overlap: a" in text
    assert "(0,)" not in text
    # Each relatum names its distinction's mechanism, direction, and the
    # state-cased purview label (read off the endpoints).
    assert "relata" in text
    for i in face.endpoints:
        ep = xor_projection.endpoints[i]
        assert ep.label in text
        assert ep.direction in text


def test_render_hub_hover_is_rich(xor_projection):
    fig = _render(xor_projection)
    hub = next(t for t in fig.data if getattr(t, "mode", None) == "markers")
    # Every hub carries the rich multi-line hover (overlap + relata), not the
    # bare fallback.
    assert all("relata" in h and "overlap:" in h for h in hub.hovertext)


def test_endpoint_hover_names_distinction_and_sibling(xor_projection):
    from pyphi.visualize.render.simplicial_complex import _endpoint_hover_fn

    hover = _endpoint_hover_fn(xor_projection)
    # Endpoint 0 is the cause of distinction ab (cause purview abc, effect c).
    cause = xor_projection.endpoints[0]
    text = hover(cause)
    assert cause.label in text and "(cause)" in text and "φ =" in text
    # Names the parent distinction and its φ_d.
    node = xor_projection.nodes[cause.distinction_id]
    assert f"distinction {node.label}" in text and "φ_d =" in text
    # Names the opposite side's purview (the effect sibling).
    sibling = xor_projection.endpoints[1]
    assert f"effect: {sibling.label}" in text


def test_distinction_hover_names_both_purviews(xor_projection):
    from pyphi.visualize.render.simplicial_complex import _distinction_hover_fn

    hover = _distinction_hover_fn(xor_projection)
    node = xor_projection.nodes[0]
    text = hover(node)
    assert f"<b>{node.label}</b>" in text and "φ_d =" in text
    # Cause and effect purviews, each with its own φ, and the total relation φ.
    assert f"cause: {xor_projection.endpoints[0].label}" in text
    assert f"effect: {xor_projection.endpoints[1].label}" in text
    assert "Σφ_r =" in text


def test_render_distinction_hovers_are_rich(xor_projection):
    fig = _render(xor_projection)
    purviews, mechanisms = fig.data[0], fig.data[1]
    # Purview dots name their parent distinction; mechanism labels name both
    # purviews and the relation total.
    assert all("distinction" in h for h in purviews.hovertext)
    assert all("Σφ_r =" in h for h in mechanisms.hovertext)


def test_render_includes_higher_face_stars(xor_projection):
    import plotly.graph_objects as go

    # star_min_degree=4 keeps degree-2 lines and degree-3 mesh, so the stars
    # cover only degree->=4 faces and land after the six base element traces.
    fig = _render(xor_projection, star_min_degree=4)
    assert len(fig.data) == 8
    hub, spokes = fig.data[6], fig.data[7]
    assert isinstance(hub, go.Scatter3d) and hub.mode == "markers"
    assert isinstance(spokes, go.Scatter3d) and spokes.mode == "lines"
    # 35 + 16 + 3 = 54 degree->=4 faces, one hub each.
    assert len(hub.x) == 54


def test_render_degrees_filter(xor_projection):
    import plotly.graph_objects as go

    # With star_min_degree=4, degrees=(2, 3) draws the geometric forms only
    # (line + mesh): 6 base traces, no stars.
    low = _render(xor_projection, degrees=(2, 3), star_min_degree=4)
    assert len(low.data) == 6
    # degrees=(4, 5, 6) keeps base elements + stars, no two/three faces.
    high = _render(xor_projection, degrees=(4, 5, 6), star_min_degree=4)
    assert not any(isinstance(t, go.Mesh3d) for t in high.data)  # no degree-3 mesh
    assert any(getattr(t, "mode", None) == "markers" for t in high.data)  # hubs


def test_render_higher_faces_show_toggle(xor_projection):
    # star_min_degree=4 so two_faces/three_faces populate; omitting
    # higher_faces from show then drops the star traces.
    fig = _render(
        xor_projection,
        show=("purviews", "two_faces", "three_faces"),
        star_min_degree=4,
    )
    assert len(fig.data) == 3


def test_star_min_degree_two_draws_every_face_as_a_star(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection, star_min_degree=2)
    # No geometric forms: no degree-2 line trace, no degree-3 mesh.
    assert not any(isinstance(t, go.Mesh3d) for t in fig.data)
    # Four base traces + the star hub + spokes.
    assert len(fig.data) == 6
    hub = next(t for t in fig.data if getattr(t, "mode", None) == "markers")
    # Every face (25 + 40 + 35 + 16 + 3 = 119) is now a hub.
    assert len(hub.x) == 119


def test_star_min_degree_three_keeps_lines_stars_the_rest(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection, star_min_degree=3)
    # Degree-2 lines remain; degree-3 is no longer a mesh.
    assert not any(isinstance(t, go.Mesh3d) for t in fig.data)
    hub = next(t for t in fig.data if getattr(t, "mode", None) == "markers")
    # 40 + 35 + 16 + 3 = 94 faces of degree >= 3 are hubs.
    assert len(hub.x) == 94


def test_star_min_degree_out_of_range_raises(xor_projection):
    with pytest.raises(ValueError, match="star_min_degree must be 2, 3, or 4"):
        _render(xor_projection, star_min_degree=5)
    with pytest.raises(ValueError, match="star_min_degree must be 2, 3, or 4"):
        _render(xor_projection, star_min_degree=1)


def test_plot_ces_star_min_degree_plumbed():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_ces

    ces = examples.xor_system().ces()
    fig = plot_ces(ces, view="simplicial_complex", star_min_degree=2)
    assert not any(isinstance(t, go.Mesh3d) for t in fig.data)


def test_render_show_subsetting(xor_projection):
    fig = _render(xor_projection, show=("purviews",))
    assert len(fig.data) == 1
    fig = _render(xor_projection, show=("purviews", "three_faces"), star_min_degree=4)
    assert len(fig.data) == 2
    with pytest.raises(ValueError, match="show"):
        _render(xor_projection, show=("purviews", "bogus"))


def test_render_only_distinctions_filters_without_moving(xor_projection):
    # star_min_degree=4 so the degree-3 mesh exists to compare.
    full = _render(xor_projection, star_min_degree=4)
    sub = _render(xor_projection, only_distinctions={0, 3}, star_min_degree=4)
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


def test_plot_ces_simplicial_complex_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_ces
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    ces = examples.xor_system().ces()
    # Default is all-stars: four base traces + the star hub + spokes.
    fig = plot_ces(ces, view="simplicial_complex")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6
    fig = plot_ces(
        ces,
        view="simplicial_complex",
        geometry=SimplicialComplexGeometry(z_spacing=0.3),
        show=("purviews",),
    )
    assert len(fig.data) == 1
    # The shared layout knob applies to this view too.
    for layout in ("barycentric", "sorted"):
        fig = plot_ces(ces, view="simplicial_complex", layout=layout)
        assert len(fig.data) == 6


def test_highlight_phi_fold_smoke():
    from types import SimpleNamespace

    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = SimpleNamespace(distinctions=list(ces.distinctions)[:2])
    fig = highlight_phi_fold(ces, fold)
    # Two all-stars passes of six traces each: dimmed full structure (indices
    # 0-5) + highlighted two-distinction fold (indices 6-11).
    assert len(fig.data) == 12
    # The overlay's endpoint coordinates are a subset of the background's.
    bg, overlay = fig.data[0], fig.data[6]
    bg_points = set(zip(bg.x, bg.y, bg.z, strict=True))
    overlay_points = set(zip(overlay.x, overlay.y, overlay.z, strict=True))
    assert len(overlay.x) == 4
    assert overlay_points <= bg_points


def test_render_colorbars(xor_projection):
    import plotly.graph_objects as go

    # star_min_degree=4 gives the full line + mesh + hub layout.
    fig = _render(xor_projection, star_min_degree=4)
    purviews = fig.data[0]
    mesh = next(t for t in fig.data if isinstance(t, go.Mesh3d))
    two_faces = fig.data[4]
    hub = fig.data[6]
    # Two colorbars total: distinction phi on the purviews, and one shared
    # "relation phi" bar drawn by the first relation class (the 2-faces). The
    # 3-face mesh and the >=4-face hubs share that scale but draw no bar.
    assert purviews.marker.showscale
    assert two_faces.line.showscale
    assert not mesh.showscale
    assert not hub.marker.showscale
    assert purviews.marker.colorbar.title.text == "φ"
    assert two_faces.line.colorbar.title.text == "relation φ"
    # The relation classes share one color range (absolute phi across degrees).
    assert two_faces.line.cmin == mesh.cmin == hub.marker.cmin
    assert two_faces.line.cmax == mesh.cmax == hub.marker.cmax
    # Opt out.
    fig = _render(xor_projection, show_colorbars=False, star_min_degree=4)
    assert not fig.data[0].marker.showscale
    assert not fig.data[4].line.showscale
    mesh = next(t for t in fig.data if isinstance(t, go.Mesh3d))
    assert not mesh.showscale


def test_highlight_phi_fold_dimmed_pass_has_no_colorbars():
    from types import SimpleNamespace

    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = SimpleNamespace(distinctions=list(ces.distinctions)[:2])
    fig = highlight_phi_fold(ces, fold)
    background, overlay = fig.data[0], fig.data[6]
    assert not background.marker.showscale
    assert overlay.marker.showscale


def test_plot_ces_rejects_phi_fold():
    from pyphi import examples
    from pyphi.visualize import plot_ces

    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    with pytest.raises(TypeError, match="highlight_phi_fold"):
        plot_ces(fold)


def test_highlight_phi_fold_one_argument():
    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    figure = highlight_phi_fold(fold)
    assert figure is not None
    assert len(figure.data) > 0


def test_highlight_phi_fold_two_argument_still_works():
    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    figure = highlight_phi_fold(ces, fold)
    assert figure is not None
