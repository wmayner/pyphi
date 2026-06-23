"""Relational-role scatter renderer for CES projections.

Positions come from a deterministic PCA of each distinction's composition
vector (mechanism, cause purview, and effect purview unit memberships) — a
reproducible stand-in for the t-SNE composition embedding of Haun & Tononi
2019 (Figs 7-8). Roles are derived from the projection's purview-union
inclusion flags (the closest computed structure to the paper's
relation-defined extendedness).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.render.common import CHANNEL_TITLES
from pyphi.visualize.render.common import rescale
from pyphi.visualize.render.common import spread_coincident
from pyphi.visualize.render.embedding import pca_embed
from pyphi.visualize.theme import Theme


def _pca_coords(projection: CESProjection) -> list[tuple[float, float]]:
    """First two principal components of distinction composition.

    Each distinction's vector concatenates the unit memberships of its
    mechanism, cause purview, and effect purview, so no two distinctions
    share a vector (mechanisms are unique). Components are sign-fixed
    (largest-magnitude loading positive); zero-variance components fall
    back to spreading nodes evenly by id, and points that still project
    onto the same spot are spread apart on a small circle.
    """
    units = sorted(
        {
            u
            for n in projection.nodes
            for u in (*n.mechanism, *n.cause_purview, *n.effect_purview)
        }
    )
    column = {u: k for k, u in enumerate(units)}
    width = len(units)
    members = np.zeros((len(projection.nodes), 3 * width))
    for n in projection.nodes:
        subsets = (n.mechanism, n.cause_purview, n.effect_purview)
        for block, subset in enumerate(subsets):
            for u in subset:
                members[n.id, block * width + column[u]] = 1.0
    coords = pca_embed(members, n_components=2)
    span = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1]))) or 1.0
    spread = spread_coincident(coords, 0.03 * span)
    return [(float(x), float(y)) for x, y in spread]


def _role(node) -> str:
    if node.includes and node.included:
        return "extended"
    if node.includes:
        return "includes"
    if node.included:
        return "included"
    return "none"


def _connected(projection: CESProjection) -> set[int]:
    """Ids of distinctions related to at least one other distinction."""
    connected: set[int] = set()
    for e in projection.edges:
        relata = set(e.relata)
        if len(relata) > 1:
            connected |= relata
    return connected


def render_scatter(
    projection: CESProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    size_by: str | None = "sum_phi_relations",
    color_by: str = "role",
) -> go.Figure:
    """Scatter distinctions by composition, encoding relational roles.

    Marker size encodes ``size_by``; color encodes the relational-role
    category (``color_by="role"``) or a numeric channel; circles mark
    distinctions related to at least one other distinction, open diamonds
    those that only self-relate.
    """
    if size_by is not None and size_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown size_by {size_by!r}")
    if color_by != "role" and color_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown color_by {color_by!r}")
    nodes = projection.nodes
    coords = _pca_coords(projection)
    connected = _connected(projection)
    roles = [_role(n) for n in nodes]
    if color_by == "role":
        palette = dict(theme.role_colors)
        marker_color = {"color": [palette[r] for r in roles]}
    else:
        marker_color = {
            "color": [getattr(n, color_by) for n in nodes],
            "colorscale": theme.colorscale,
            "colorbar": {"title": CHANNEL_TITLES[color_by]},
        }
    smin, smax = theme.node_size_range
    sizes = (
        [(smin + smax) / 2.0] * len(nodes)
        if size_by is None
        else rescale([getattr(n, size_by) for n in nodes], smin, smax)
    )
    hover = [
        (
            f"<b>{n.label}</b> ({role})"
            f"<br>mechanism {n.mechanism} = {n.mechanism_state}"
            f"<br>cause {n.cause_purview} · effect {n.effect_purview}"
            f"<br>φ = {n.phi:.4g} · Σφ_R = {n.sum_phi_relations:.4g}"
        )
        for n, role in zip(nodes, roles, strict=True)
    ]
    trace = go.Scatter(
        x=[coords[n.id][0] for n in nodes],
        y=[coords[n.id][1] for n in nodes],
        mode="markers+text",
        text=[n.label for n in nodes],
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker={
            "size": sizes,
            "symbol": ["circle" if n.id in connected else "diamond-open" for n in nodes],
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
            **marker_color,
        },
        showlegend=False,
    )
    figure = go.Figure() if fig is None else fig
    figure.add_trace(trace)
    figure.update_layout(
        plot_bgcolor=theme.background,
        font={"family": theme.font_family},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
