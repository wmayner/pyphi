"""3-D simplicial-complex renderer for CES projections."""

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
class SimplicialComplexGeometry:
    """Plot-space layout knobs for the simplicial-complex view.

    ``z_spacing`` stacks the size shells vertically, so height encodes
    subset size; set it to 0 for a flat (planar) arrangement.
    """

    max_radius: float = 1.0
    z_spacing: float = 0.4
    direction_offset: float = 0.3
    purview_jitter: float = 0.1


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
    rings: Rings, geometry: SimplicialComplexGeometry
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
    subsets: Iterable[tuple[int, ...]], geometry: SimplicialComplexGeometry
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


def _endpoint_positions(
    projection: CESProjection,
    geometry: SimplicialComplexGeometry,
    base: dict[tuple[int, ...], Point] | None = None,
) -> dict[int, Point]:
    """Position each endpoint near its purview's shell point.

    Cause endpoints shift -x and effect endpoints +x by
    ``direction_offset``; endpoints sharing a purview and direction spread
    on a small polygon of radius ``purview_jitter``.
    """
    if base is None:
        base = _shell_positions((e.purview for e in projection.endpoints), geometry)
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
        offsets = _polygon_points(len(ids), jitter, 0.0)
        for eid, (ox, oy, _) in zip(sorted(ids), offsets, strict=True):
            positions[eid] = (bx + ox, by + oy, bz)
    return positions


def _mechanism_positions(
    projection: CESProjection,
    geometry: SimplicialComplexGeometry,
    base: dict[tuple[int, ...], Point] | None = None,
) -> dict[int, Point]:
    """Position each distinction's mechanism on its size shell."""
    if base is None:
        base = _shell_positions((n.mechanism for n in projection.nodes), geometry)
    return {n.id: base[n.mechanism] for n in projection.nodes}


def _positions_3d(
    projection: CESProjection,
    geometry: SimplicialComplexGeometry,
    layout: str = "barycentric",
) -> tuple[dict[int, Point], dict[int, Point]]:
    """Endpoint and mechanism positions under the chosen layout.

    ``layout="sorted"`` orders each shell ring lexicographically.
    ``layout="barycentric"`` reorders purview rings so subsets connected
    by drawn elements sit at nearby angles, then orders each mechanism
    ring by the mean angle of its distinction's purviews.
    """
    if layout not in ("barycentric", "sorted"):
        raise ValueError(f"unknown layout {layout!r}")
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
    endpoint_pos = _endpoint_positions(
        projection, geometry, base=_positions_from_rings(purview_rings, geometry)
    )
    mechanism_pos = _mechanism_positions(
        projection, geometry, base=_positions_from_rings(mechanism_rings, geometry)
    )
    return endpoint_pos, mechanism_pos


_ELEMENTS = (
    "purviews",
    "mechanisms",
    "cause_effect_links",
    "mechanism_purview_links",
    "two_faces",
    "three_faces",
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


def _purview_trace(endpoints, pos, theme):
    hover = [
        (
            f"<b>{e.label}</b> ({e.direction})"
            f"<br>purview {e.purview} = {e.purview_state}"
            f"<br>φ = {e.phi:.4g}"
        )
        for e in endpoints
    ]
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
            "showscale": False,
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
        },
        showlegend=False,
    )


def _mechanism_trace(nodes, pos, theme):
    hover = [
        f"<b>{n.label}</b><br>mechanism {n.mechanism} = {n.mechanism_state}"
        f"<br>φ = {n.phi:.4g}"
        for n in nodes
    ]
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


def _two_face_trace(faces, endpoint_pos, theme):
    xs, ys, zs = _segments(tuple(endpoint_pos[i] for i in f.endpoints) for f in faces)
    # One color value per vertex, including the None separators.
    colors = [phi for f in faces for phi in [f.phi] * 3]
    hover = [
        f"2-face<br>overlap {f.overlap}<br>φ = {f.phi:.4g}"
        for f in faces
        for _ in range(3)
    ]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={
            "color": colors,
            "colorscale": theme.face_colorscale,
            "width": 2 * theme.edge_width,
        },
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _three_face_trace(faces, endpoint_pos, theme):
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
        colorscale=theme.face_colorscale,
        opacity=theme.face_opacity,
        showscale=False,
        hoverinfo="skip",
    )


def render_simplicial_complex(
    projection: CESProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    geometry: SimplicialComplexGeometry | None = None,
    show: tuple[str, ...] = _ELEMENTS,
    only_distinctions: set[int] | None = None,
    layout: str = "barycentric",
) -> go.Figure:
    """Draw the cause-effect structure as a 3-D simplicial complex.

    Purview endpoints are vertices; degree-2 relation faces are line
    segments and degree-3 faces are triangles. Geometry is computed from
    the full projection regardless of ``only_distinctions``, so successive
    calls with different subsets align (the primitive
    ``highlight_phi_fold`` composes on).
    """
    unknown = set(show) - set(_ELEMENTS)
    if unknown:
        raise ValueError(f"unknown show element(s) {sorted(unknown)!r}")
    if geometry is None:
        geometry = SimplicialComplexGeometry()
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
    two_faces = [f for f in faces if f.degree == 2]
    three_faces = [f for f in faces if f.degree == 3]
    traces = []
    if "purviews" in show:
        traces.append(_purview_trace(endpoints, endpoint_pos, theme))
    if "mechanisms" in show:
        traces.append(_mechanism_trace(nodes, mechanism_pos, theme))
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
        traces.append(_two_face_trace(two_faces, endpoint_pos, theme))
    if "three_faces" in show and three_faces:
        traces.append(_three_face_trace(three_faces, endpoint_pos, theme))
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
