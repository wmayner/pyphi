"""3-D hypergraph renderer for CES projections."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.render.common import rescale
from pyphi.visualize.theme import Theme

Point = tuple[float, float, float]


@dataclass(frozen=True)
class HypergraphGeometry:
    """Plot-space layout knobs for the hypergraph view.

    ``z_spacing`` stacks the size shells vertically, so height encodes
    subset size; set it to 0 for a flat (planar) arrangement.
    """

    max_radius: float = 1.0
    z_spacing: float = 0.4
    direction_offset: float = 0.3
    purview_jitter: float = 0.1
    endpoint_placement: str = "mechanism_anchored"
    embedding_method: str = "mds"


# Endpoints sharing a purview are nudged toward their mechanism; vectors shorter
# than this (mechanism coincident with the shell point in xy) carry no direction
# and fall to the polygon tie-break, as do any that land on the same quantized
# angle (collinear mechanisms).
_ANCHOR_EPS = 1e-12
_ANCHOR_ANGLE_GRID = 1e-6


def _polygon_points(n: int, radius: float, z: float) -> list[Point]:
    """``n`` points evenly spaced on a circle of ``radius`` at height ``z``."""
    return [
        (
            radius * math.cos(2 * math.pi * k / n),
            radius * math.sin(2 * math.pi * k / n),
            z,
        )
        for k in range(n)
    ]


Rings = dict[int, list[tuple[int, ...]]]

_N_ANGULAR_SWEEPS = 4


def _rings(subsets: Iterable[tuple[int, ...]]) -> Rings:
    """Group unique subsets into size rings, members in sorted order."""
    by_size: Rings = defaultdict(list)
    for s in sorted(set(subsets)):
        by_size[len(s)].append(s)
    return dict(by_size)


def _positions_from_rings(
    rings: Rings, geometry: HypergraphGeometry
) -> dict[tuple[int, ...], Point]:
    """Place each subset on the shell for its size, in ring order.

    Subsets of size k share a circular shell whose radius grows linearly
    with k up to ``max_radius``; within a shell, subsets sit on a regular
    polygon in the given ring order, and a shell with a single subset sits
    on the central axis. Shells stack in z by ``z_spacing``.
    """
    sizes = sorted(rings)
    k_max = max(sizes)
    positions: dict[tuple[int, ...], Point] = {}
    for shell_index, k in enumerate(sizes):
        members = rings[k]
        radius = geometry.max_radius * k / k_max
        z = geometry.z_spacing * shell_index
        points = (
            [(0.0, 0.0, z)]
            if len(members) == 1
            else _polygon_points(len(members), radius, z)
        )
        positions.update(zip(members, points, strict=True))
    return positions


def _shell_positions(
    subsets: Iterable[tuple[int, ...]], geometry: HypergraphGeometry
) -> dict[tuple[int, ...], Point]:
    """Place each unique subset on the shell for its size, sorted order."""
    return _positions_from_rings(_rings(subsets), geometry)


def _ring_angles(rings: Rings) -> dict[tuple[int, ...], float]:
    """Polygon angle of each subset on a multi-member ring."""
    angles: dict[tuple[int, ...], float] = {}
    for members in rings.values():
        if len(members) > 1:
            for k, s in enumerate(members):
                angles[s] = 2 * math.pi * k / len(members)
    return angles


def _circular_mean(weighted_angles: list[tuple[float, float]]) -> float | None:
    """Mean direction of (angle, weight) pairs; None if they cancel."""
    x = sum(w * math.cos(a) for a, w in weighted_angles)
    y = sum(w * math.sin(a) for a, w in weighted_angles)
    if math.hypot(x, y) < 1e-12:
        return None
    return math.atan2(y, x) % (2 * math.pi)


def _subset_graph(
    projection: CESProjection,
) -> dict[tuple[int, ...], dict[tuple[int, ...], float]]:
    """Weights between purview subsets connected by drawn elements.

    Relation faces and cause-effect links are the edges actually drawn
    between endpoints; each contributes weight between the (distinct)
    purview subsets of its endpoints.
    """
    graph: dict[tuple[int, ...], dict[tuple[int, ...], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    def add(a: tuple[int, ...], b: tuple[int, ...]) -> None:
        if a != b:
            graph[a][b] += 1.0
            graph[b][a] += 1.0

    for face in projection.faces:
        for i, j in combinations(face.endpoints, 2):
            add(projection.endpoints[i].purview, projection.endpoints[j].purview)
    for node in projection.nodes:
        add(
            projection.endpoints[2 * node.id].purview,
            projection.endpoints[2 * node.id + 1].purview,
        )
    return graph


def _reorder_rings(
    rings: Rings,
    graph: dict[tuple[int, ...], dict[tuple[int, ...], float]],
) -> Rings:
    """Reorder each ring to put connected subsets at nearby angles.

    Repeated anchored sweeps: each member's target angle is the circular
    mean of its graph neighbors' current angles (updated sequentially in
    sorted order for determinism), then the ring is re-spread evenly in
    target order. An interim heuristic that shortens drawn edges; not a
    crossing-minimizer.
    """
    rings = {size: list(members) for size, members in rings.items()}
    angles = _ring_angles(rings)
    for _ in range(_N_ANGULAR_SWEEPS):
        for size in sorted(rings):
            members = rings[size]
            if len(members) < 2:
                continue
            targets: dict[tuple[int, ...], float] = {}
            for s in sorted(members):
                weighted = [
                    (angles[t], w) for t, w in graph.get(s, {}).items() if t in angles
                ]
                mean = _circular_mean(weighted) if weighted else None
                targets[s] = angles[s] if mean is None else mean
                angles[s] = targets[s]
            members = sorted(members, key=lambda s: (targets[s], s))
            rings[size] = members
            for k, s in enumerate(members):
                angles[s] = 2 * math.pi * k / len(members)
    return rings


def _anchored_offsets(ids, base_xy, mechanism_pos, projection, jitter):
    """xy offsets that nudge each endpoint toward its mechanism.

    Each endpoint moves ``jitter`` from the shell point in the direction of its
    distinction's mechanism. Endpoints whose mechanism direction is degenerate
    (mechanism coincident with the shell point in xy) or collinear with another
    member's (same quantized angle) carry no usable direction; those residual
    subgroups fall back to the even polygon spread, so the result is always
    collision-free and deterministic.
    """
    bx, by = base_xy
    direction: dict[int, float | None] = {}
    for eid in ids:
        mx, my, _ = mechanism_pos[projection.endpoints[eid].distinction_id]
        dx, dy = mx - bx, my - by
        norm = math.hypot(dx, dy)
        direction[eid] = math.atan2(dy, dx) if norm > _ANCHOR_EPS else None
    # Group by quantized angle; a unique angle keeps its mechanism direction,
    # while ties (and the directionless degenerate ones) are spread evenly.
    by_angle: dict[float | None, list[int]] = defaultdict(list)
    for eid in sorted(ids):
        angle = direction[eid]
        key = None if angle is None else round(angle / _ANCHOR_ANGLE_GRID)
        by_angle[key].append(eid)
    offsets: dict[int, tuple[float, float]] = {}
    for key, group in by_angle.items():
        angle = direction[group[0]] if key is not None else None
        if angle is not None and len(group) == 1:
            offsets[group[0]] = (jitter * math.cos(angle), jitter * math.sin(angle))
        else:
            for point, eid in zip(
                _polygon_points(len(group), jitter, 0.0), group, strict=True
            ):
                offsets[eid] = (point[0], point[1])
    return offsets


def _endpoint_positions(
    projection: CESProjection,
    geometry: HypergraphGeometry,
    base: dict[tuple[int, ...], Point] | None = None,
    mechanism_pos: dict[int, Point] | None = None,
) -> dict[int, Point]:
    """Position each endpoint near its purview's shell point.

    Cause endpoints shift -x and effect endpoints +x by ``direction_offset``.
    Endpoints sharing a purview and direction are separated by
    ``endpoint_placement``: ``"mechanism_anchored"`` nudges each toward its
    distinction's mechanism (so the dot sits on its mechanism-purview link),
    while ``"polygon"`` spreads them on a small regular polygon of radius
    ``purview_jitter``. Both keep the shell point's z.
    """
    if geometry.endpoint_placement not in ("mechanism_anchored", "polygon"):
        raise ValueError(f"unknown endpoint_placement {geometry.endpoint_placement!r}")
    if base is None:
        base = _shell_positions((e.purview for e in projection.endpoints), geometry)
    if mechanism_pos is None and geometry.endpoint_placement == "mechanism_anchored":
        mechanism_pos = _mechanism_positions(projection, geometry)
    groups: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(list)
    for e in projection.endpoints:
        groups[(e.purview, e.direction)].append(e.id)
    positions: dict[int, Point] = {}
    for (purview, direction), ids in groups.items():
        bx, by, bz = base[purview]
        bx += (
            geometry.direction_offset
            if direction == "effect"
            else -geometry.direction_offset
        )
        jitter = geometry.purview_jitter if len(ids) > 1 else 0.0
        if geometry.endpoint_placement == "mechanism_anchored" and jitter:
            offsets = _anchored_offsets(ids, (bx, by), mechanism_pos, projection, jitter)
            for eid in ids:
                ox, oy = offsets[eid]
                positions[eid] = (bx + ox, by + oy, bz)
        else:
            polygon = _polygon_points(len(ids), jitter, 0.0)
            for eid, (ox, oy, _) in zip(sorted(ids), polygon, strict=True):
                positions[eid] = (bx + ox, by + oy, bz)
    return positions


def _mechanism_positions(
    projection: CESProjection,
    geometry: HypergraphGeometry,
    base: dict[tuple[int, ...], Point] | None = None,
) -> dict[int, Point]:
    """Position each distinction's mechanism on its size shell."""
    if base is None:
        base = _shell_positions((n.mechanism for n in projection.nodes), geometry)
    return {n.id: base[n.mechanism] for n in projection.nodes}


def _positions_3d(
    projection: CESProjection,
    geometry: HypergraphGeometry,
    layout: str = "barycentric",
) -> tuple[dict[int, Point], dict[int, Point]]:
    """Endpoint and mechanism positions under the chosen layout.

    ``layout="sorted"`` orders each shell ring lexicographically.
    ``layout="barycentric"`` reorders purview rings so subsets connected
    by drawn elements sit at nearby angles, then orders each mechanism
    ring by the mean angle of its distinction's purviews.
    """
    if layout not in ("barycentric", "sorted", "embedding"):
        raise ValueError(f"unknown layout {layout!r}")
    if layout == "embedding":
        from pyphi.visualize.render.embedding import embedding_positions

        return embedding_positions(projection, geometry)
    purview_rings = _rings(e.purview for e in projection.endpoints)
    mechanism_rings = _rings(n.mechanism for n in projection.nodes)
    if layout == "barycentric":
        purview_rings = _reorder_rings(purview_rings, _subset_graph(projection))
        purview_angles = _ring_angles(purview_rings)
        targets: dict[tuple[int, ...], float] = {}
        for node in projection.nodes:
            weighted = [
                (purview_angles[projection.endpoints[eid].purview], 1.0)
                for eid in (2 * node.id, 2 * node.id + 1)
                if projection.endpoints[eid].purview in purview_angles
            ]
            mean = _circular_mean(weighted) if weighted else None
            if mean is not None:
                targets[node.mechanism] = mean
        mechanism_rings = {
            size: sorted(members, key=lambda s: (targets.get(s, 0.0), s))
            for size, members in mechanism_rings.items()
        }
    mechanism_pos = _mechanism_positions(
        projection, geometry, base=_positions_from_rings(mechanism_rings, geometry)
    )
    endpoint_pos = _endpoint_positions(
        projection,
        geometry,
        base=_positions_from_rings(purview_rings, geometry),
        mechanism_pos=mechanism_pos,
    )
    return endpoint_pos, mechanism_pos


_ELEMENTS = (
    "purviews",
    "mechanisms",
    "cause_effect_links",
    "mechanism_purview_links",
    "two_faces",
    "three_faces",
    "higher_faces",
)


def _segments(
    paths: Iterable[tuple[Point, ...]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """None-separated coordinate arrays from point paths."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for path in paths:
        for x, y, z in path:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return xs, ys, zs


# Points sampled along each curved spoke; higher is smoother but heavier.
_SPOKE_ARC_SEGMENTS = 8


def _unit_perp(d: Point) -> Point:
    """A unit vector perpendicular to ``d`` (deterministic; zero if ``d`` is)."""
    dx, dy, dz = d
    ref = (0.0, 1.0, 0.0) if abs(dx) < 1e-9 and abs(dy) < 1e-9 else (0.0, 0.0, 1.0)
    px = dy * ref[2] - dz * ref[1]
    py = dz * ref[0] - dx * ref[2]
    pz = dx * ref[1] - dy * ref[0]
    norm = math.sqrt(px * px + py * py + pz * pz)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return (px / norm, py / norm, pz / norm)


def _arc(p0: Point, p1: Point, curvature: float) -> tuple[Point, ...]:
    """Sample a quadratic-Bezier arc from ``p0`` to ``p1``, bowed perpendicular
    to the chord by ``curvature`` times its length. Straight (the two
    endpoints) when ``curvature`` is zero.
    """
    if curvature <= 0.0:
        return (p0, p1)
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    perp = _unit_perp(d)
    off = curvature * length
    ctrl = (
        (p0[0] + p1[0]) / 2 + perp[0] * off,
        (p0[1] + p1[1]) / 2 + perp[1] * off,
        (p0[2] + p1[2]) / 2 + perp[2] * off,
    )
    points = []
    for step in range(_SPOKE_ARC_SEGMENTS + 1):
        t = step / _SPOKE_ARC_SEGMENTS
        u = 1.0 - t
        a, b, c = u * u, 2 * u * t, t * t
        points.append(
            (
                a * p0[0] + b * ctrl[0] + c * p1[0],
                a * p0[1] + b * ctrl[1] + c * p1[1],
                a * p0[2] + b * ctrl[2] + c * p1[2],
            )
        )
    return tuple(points)


def _basic_endpoint_hover(e):
    """Fallback hover for a purview endpoint when no projection is supplied."""
    return f"<b>{e.label}</b> ({e.direction}) · φ = {e.phi:.4g}"


def _basic_distinction_hover(n):
    """Fallback hover for a distinction when no projection is supplied."""
    return f"<b>{n.label}</b> · φ_d = {n.phi:.4g}"


def _purview_trace(
    endpoints, pos, theme, show_colorbar=True, hover_for=_basic_endpoint_hover
):
    hover = [hover_for(e) for e in endpoints]
    return go.Scatter3d(
        x=[pos[e.id][0] for e in endpoints],
        y=[pos[e.id][1] for e in endpoints],
        z=[pos[e.id][2] for e in endpoints],
        mode="markers+text",
        text=[e.label for e in endpoints],
        textposition="top center",
        textfont={
            "size": theme.text_size,
            "color": [
                theme.cause_color if e.direction == "cause" else theme.effect_color
                for e in endpoints
            ],
        },
        hovertext=hover,
        hoverinfo="text",
        marker={
            "size": rescale([e.phi for e in endpoints], *theme.node_size_range),
            "color": [e.phi for e in endpoints],
            "colorscale": theme.colorscale,
            "showscale": show_colorbar,
            "colorbar": {"title": "φ", "x": 1.02, "len": 0.6},
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
        },
        showlegend=False,
    )


def _mechanism_trace(nodes, pos, theme, hover_for=_basic_distinction_hover):
    hover = [hover_for(n) for n in nodes]
    return go.Scatter3d(
        x=[pos[n.id][0] for n in nodes],
        y=[pos[n.id][1] for n in nodes],
        z=[pos[n.id][2] for n in nodes],
        mode="text",
        text=[n.label for n in nodes],
        textfont={"size": theme.text_size},
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _link_trace(paths, theme):
    xs, ys, zs = _segments(paths)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": theme.edge_color, "width": 2 * theme.edge_width},
        hoverinfo="skip",
        showlegend=False,
    )


_RELATION_COLORBAR = {"title": "relation φ", "x": 1.14, "len": 0.6}


def _relation_colorscale(theme):
    """A fixed-hue scale whose opacity (not lightness) tracks φ, so low-φ
    relations fade toward transparency rather than toward a pale colour that
    would compete with the neutral grey spokes.
    """
    r, g, b = theme.relation_rgb
    lo, hi = theme.relation_alpha_range
    return [[0.0, f"rgba({r}, {g}, {b}, {lo})"], [1.0, f"rgba({r}, {g}, {b}, {hi})"]]


def _basic_face_hover(face):
    """Fallback hover when no endpoint/label lookups are supplied."""
    return f"{face.degree}-face · φ = {face.phi:.4g}"


def _face_hover_fn(projection):
    """A callable giving rich hover text for a relation face: degree, φ, the
    labelled overlap units, and each relatum (its distinction's mechanism,
    direction, and state-cased purview).
    """
    endpoint_by_id = {e.id: e for e in projection.endpoints}
    node_by_did = {n.id: n for n in projection.nodes}
    labels = projection.node_labels

    def hover(face):
        # Plain (stateless) unit labels, lowercased so they are not mistaken
        # for the state-cased purview labels where upper case means ON.
        overlap = (
            "".join(labels.indices2labels(face.overlap)).lower() if face.overlap else "∅"
        )
        relata = "<br>".join(
            f"  {node_by_did[ep.distinction_id].label}: {ep.direction} {ep.label}"
            for ep in (endpoint_by_id[i] for i in face.endpoints)
        )
        return (
            f"<b>{face.degree}-face</b> · φ = {face.phi:.4g}"
            f"<br>overlap: {overlap}"
            f"<br>relata:<br>{relata}"
        )

    return hover


def _endpoint_hover_fn(projection):
    """Rich hover for a purview endpoint (one side of a distinction): the
    state-cased purview and direction, this side's φ, and the parent
    distinction (mechanism, φ_d, and the opposite purview).
    """
    nodes_by_id = {n.id: n for n in projection.nodes}
    endpoints = projection.endpoints

    def hover(e):
        node = nodes_by_id[e.distinction_id]
        sibling = endpoints[e.id ^ 1]
        other = "effect" if e.direction == "cause" else "cause"
        return (
            f"<b>{e.label}</b> ({e.direction}) · φ = {e.phi:.4g}"
            f"<br>distinction {node.label} · φ_d = {node.phi:.4g}"
            f"<br>{other}: {sibling.label}"
        )

    return hover


def _distinction_hover_fn(projection):
    """Rich hover for a distinction (its mechanism marker): the mechanism, φ_d,
    the cause and effect purviews (state-cased, each with its φ), and the total
    relation φ the distinction carries.
    """
    endpoints = projection.endpoints

    def hover(node):
        cause = endpoints[2 * node.id]
        effect = endpoints[2 * node.id + 1]
        return (
            f"<b>{node.label}</b> · φ_d = {node.phi:.4g}"
            f"<br>cause: {cause.label} · φ = {cause.phi:.4g}"
            f"<br>effect: {effect.label} · φ = {effect.phi:.4g}"
            f"<br>Σφ_r = {node.sum_phi_relations:.4g}"
        )

    return hover


def _two_face_trace(
    faces,
    endpoint_pos,
    theme,
    show_colorbar=True,
    cmin=None,
    cmax=None,
    hover_for_face=_basic_face_hover,
):
    xs, ys, zs = _segments(tuple(endpoint_pos[i] for i in f.endpoints) for f in faces)
    # One color value per vertex, including the None separators.
    colors = [phi for f in faces for phi in [f.phi] * 3]
    hover = [hover_for_face(f) for f in faces for _ in range(3)]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={
            "color": colors,
            "colorscale": _relation_colorscale(theme),
            "cmin": cmin,
            "cmax": cmax,
            "width": 2 * theme.edge_width,
            "showscale": show_colorbar,
            "colorbar": _RELATION_COLORBAR,
        },
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _three_face_trace(
    faces, endpoint_pos, theme, show_colorbar=True, cmin=None, cmax=None
):
    n = max(endpoint_pos) + 1
    xs = [endpoint_pos[i][0] for i in range(n)]
    ys = [endpoint_pos[i][1] for i in range(n)]
    zs = [endpoint_pos[i][2] for i in range(n)]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=[f.endpoints[0] for f in faces],
        j=[f.endpoints[1] for f in faces],
        k=[f.endpoints[2] for f in faces],
        intensity=[f.phi for f in faces],
        intensitymode="cell",
        colorscale=_relation_colorscale(theme),
        cmin=cmin,
        cmax=cmax,
        opacity=theme.face_opacity,
        showscale=show_colorbar,
        colorbar=_RELATION_COLORBAR,
        hoverinfo="skip",
    )


def _higher_face_trace(
    faces,
    endpoint_pos,
    theme,
    show_colorbar=True,
    cmin=None,
    cmax=None,
    hover_for_face=_basic_face_hover,
):
    """Star expansion for a face of any degree: a hub at the face's centroid,
    sized and colored by phi (opacity tracking phi, so low-phi hubs fade), with
    neutral grey spokes to each endpoint (straight by default; bowed when
    ``theme.spoke_curvature`` is set). Two merged traces (the hub markers and
    the spoke lines).
    """
    hubs = []
    spoke_paths = []
    for face in faces:
        coords = [endpoint_pos[i] for i in face.endpoints]
        hub = (
            sum(c[0] for c in coords) / len(coords),
            sum(c[1] for c in coords) / len(coords),
            sum(c[2] for c in coords) / len(coords),
        )
        hubs.append((hub, face))
        spoke_paths.extend(_arc(hub, c, theme.spoke_curvature) for c in coords)
    hub_trace = go.Scatter3d(
        x=[h[0] for h, _ in hubs],
        y=[h[1] for h, _ in hubs],
        z=[h[2] for h, _ in hubs],
        mode="markers",
        marker={
            "size": rescale([f.phi for _, f in hubs], *theme.hub_size_range),
            "color": [f.phi for _, f in hubs],
            "colorscale": _relation_colorscale(theme),
            "cmin": cmin,
            "cmax": cmax,
            "symbol": "diamond",
            "showscale": show_colorbar,
            "colorbar": _RELATION_COLORBAR,
        },
        hovertext=[hover_for_face(f) for _, f in hubs],
        hoverinfo="text",
        showlegend=False,
    )
    xs, ys, zs = _segments(spoke_paths)
    spoke_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": theme.spoke_color, "width": theme.spoke_width},
        hoverinfo="skip",
        showlegend=False,
    )
    return [hub_trace, spoke_trace]


def render_hypergraph(
    projection: CESProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    geometry: HypergraphGeometry | None = None,
    show: tuple[str, ...] = _ELEMENTS,
    only_distinctions: set[int] | None = None,
    layout: str = "barycentric",
    show_colorbars: bool = True,
    degrees: tuple[int, ...] | None = None,
    star_min_degree: int = 2,
) -> go.Figure:
    """Draw the cause-effect structure as a 3-D hypergraph.

    Purview endpoints are vertices. By default (``star_min_degree=2``) every
    relation face is drawn as a star expansion (a hub at the face centroid with
    spokes to each endpoint), so no degree is given special visual weight.
    Raising ``star_min_degree`` restores geometric forms for the lower degrees:
    ``3`` draws degree-2 faces as line segments, ``4`` additionally draws
    degree-3 faces as filled triangles. All relation faces share one
    ``relation φ`` colorbar, and colour opacity tracks φ so low-φ relations
    fade rather than obscure the high-φ ones. Geometry is computed from the
    full projection regardless of ``only_distinctions``, so successive calls
    with different subsets align (the primitive ``highlight_phi_fold`` composes
    on).
    """
    unknown = set(show) - set(_ELEMENTS)
    if unknown:
        raise ValueError(f"unknown show element(s) {sorted(unknown)!r}")
    if star_min_degree not in (2, 3, 4):
        raise ValueError(
            f"star_min_degree must be 2, 3, or 4 (got {star_min_degree!r}); "
            "the geometric forms only cover degree-2 lines and degree-3 meshes"
        )
    if geometry is None:
        geometry = HypergraphGeometry()
    endpoint_pos, mechanism_pos = _positions_3d(projection, geometry, layout=layout)
    included = (
        set(range(len(projection.nodes)))
        if only_distinctions is None
        else set(only_distinctions)
    )
    endpoints = [e for e in projection.endpoints if e.distinction_id in included]
    nodes = [n for n in projection.nodes if n.id in included]
    faces = [
        f
        for f in projection.faces
        if all(projection.endpoints[i].distinction_id in included for i in f.endpoints)
    ]
    if degrees is not None:
        faces = [f for f in faces if f.degree in degrees]
    two_faces = [f for f in faces if f.degree == 2 and star_min_degree > 2]
    three_faces = [f for f in faces if f.degree == 3 and star_min_degree > 3]
    higher_faces = [f for f in faces if f.degree >= star_min_degree]
    # All relation faces share one color scale, so color means absolute phi
    # across degrees; exactly one face class draws the shared colorbar.
    relation_phis = [f.phi for f in faces]
    rel_cmin = min(relation_phis) if relation_phis else None
    rel_cmax = max(relation_phis) if relation_phis else None
    if not show_colorbars:
        bar_owner = None
    elif "two_faces" in show and two_faces:
        bar_owner = "two_faces"
    elif "three_faces" in show and three_faces:
        bar_owner = "three_faces"
    elif "higher_faces" in show and higher_faces:
        bar_owner = "higher_faces"
    else:
        bar_owner = None
    hover_for_face = _face_hover_fn(projection)
    traces = []
    if "purviews" in show:
        traces.append(
            _purview_trace(
                endpoints,
                endpoint_pos,
                theme,
                show_colorbars,
                hover_for=_endpoint_hover_fn(projection),
            )
        )
    if "mechanisms" in show:
        traces.append(
            _mechanism_trace(
                nodes,
                mechanism_pos,
                theme,
                hover_for=_distinction_hover_fn(projection),
            )
        )
    if "cause_effect_links" in show:
        traces.append(
            _link_trace(
                ((endpoint_pos[2 * n.id], endpoint_pos[2 * n.id + 1]) for n in nodes),
                theme,
            )
        )
    if "mechanism_purview_links" in show:
        traces.append(
            _link_trace(
                (
                    (
                        endpoint_pos[2 * n.id],
                        mechanism_pos[n.id],
                        endpoint_pos[2 * n.id + 1],
                    )
                    for n in nodes
                ),
                theme,
            )
        )
    if "two_faces" in show and two_faces:
        traces.append(
            _two_face_trace(
                two_faces,
                endpoint_pos,
                theme,
                show_colorbar=bar_owner == "two_faces",
                cmin=rel_cmin,
                cmax=rel_cmax,
                hover_for_face=hover_for_face,
            )
        )
    if "three_faces" in show and three_faces:
        traces.append(
            _three_face_trace(
                three_faces,
                endpoint_pos,
                theme,
                show_colorbar=bar_owner == "three_faces",
                cmin=rel_cmin,
                cmax=rel_cmax,
            )
        )
    if "higher_faces" in show and higher_faces:
        traces.extend(
            _higher_face_trace(
                higher_faces,
                endpoint_pos,
                theme,
                show_colorbar=bar_owner == "higher_faces",
                cmin=rel_cmin,
                cmax=rel_cmax,
                hover_for_face=hover_for_face,
            )
        )
    figure = go.Figure() if fig is None else fig
    figure.add_traces(traces)
    axis = {"visible": False}
    figure.update_layout(
        scene={
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": axis,
            "aspectmode": "data",
        },
        paper_bgcolor=theme.background,
        font={"family": theme.font_family},
        showlegend=False,
    )
    return figure
