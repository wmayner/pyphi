"""Inclusion-lattice (Hasse) renderer for phi-structure projections."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.theme import Theme


def _positions(projection: PhiStructureProjection) -> dict[int, tuple[float, float]]:
    """x spread within each rank (label-sorted), y = inclusion rank."""
    by_rank: dict[int, list[int]] = defaultdict(list)
    for node in projection.nodes:
        by_rank[projection.inclusion.rank[node.id]].append(node.id)
    positions: dict[int, tuple[float, float]] = {}
    for rank, members in by_rank.items():
        ordered = sorted(members, key=lambda i: projection.nodes[i].label)
        width = len(ordered) - 1
        for k, i in enumerate(ordered):
            x = k - width / 2.0
            positions[i] = (x, float(rank))
    return positions


def _node_sizes(projection: PhiStructureProjection, theme: Theme) -> list[float]:
    values = [n.sum_phi_relations for n in projection.nodes]
    lo, hi = min(values), max(values)
    smin, smax = theme.node_size_range
    if hi == lo:
        return [(smin + smax) / 2.0] * len(values)
    return [smin + (v - lo) / (hi - lo) * (smax - smin) for v in values]


def render_lattice(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Draw the inclusion partial order as a 2-D Hasse diagram."""
    pos = _positions(projection)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for a, cov in enumerate(projection.inclusion.covers):
        for b in cov:
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
    hover = [
        (
            f"<b>{n.label}</b><br>mechanism {n.mechanism} = {n.mechanism_state}"
            f"<br>cause {n.cause_purview} · effect {n.effect_purview}"
            f"<br>φ = {n.phi:.4g} · Σφ_R = {n.sum_phi_relations:.4g}"
        )
        for n in projection.nodes
    ]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": theme.edge_color, "width": theme.edge_width},
        hoverinfo="skip",
        showlegend=False,
    )
    node_trace = go.Scatter(
        x=[pos[n.id][0] for n in projection.nodes],
        y=[pos[n.id][1] for n in projection.nodes],
        mode="markers+text",
        text=[n.label for n in projection.nodes],
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker={
            "size": _node_sizes(projection, theme),
            "color": [n.phi for n in projection.nodes],
            "colorscale": theme.colorscale,
            "colorbar": {"title": "φ"},
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
        },
        showlegend=False,
    )
    figure = go.Figure() if fig is None else fig
    figure.add_traces([edge_trace, node_trace])
    figure.update_layout(
        plot_bgcolor=theme.background,
        font={"family": theme.font_family},
        xaxis={"visible": False},
        yaxis={"title": "inclusion rank", "dtick": 1},
    )
    return figure
