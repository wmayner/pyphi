"""Inclusion-lattice (Hasse) renderer for phi-structure projections."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from pyphi.visualize.projection import InclusionOrder
from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.render.common import CHANNEL_TITLES
from pyphi.visualize.render.common import rescale
from pyphi.visualize.theme import Theme

_N_BARYCENTRIC_SWEEPS = 4


def _spread(order: dict[int, list[int]]) -> dict[int, tuple[float, float]]:
    """Place each rank's nodes evenly spaced and centered, y = rank."""
    positions: dict[int, tuple[float, float]] = {}
    for rank, members in order.items():
        width = len(members) - 1
        for k, i in enumerate(members):
            positions[i] = (k - width / 2.0, float(rank))
    return positions


def _positions(
    projection: PhiStructureProjection,
    inclusion: InclusionOrder,
    layout: str = "barycentric",
    rank: str = "chain",
) -> dict[int, tuple[float, float]]:
    """Node positions: y = inclusion level, x spread within each level.

    ``rank="chain"`` places each node at its longest-down-chain rank
    (compact); ``rank="size"`` at the cardinality of its unit set, leaving
    gaps at sizes with no distinctions. ``layout="sorted"`` orders each
    level by label. ``layout="barycentric"`` starts from label order, then
    repeatedly reorders each level by the mean x of each node's cover
    neighbors, reducing edge crossings.
    """
    if layout not in ("barycentric", "sorted"):
        raise ValueError(f"unknown layout {layout!r}")
    if rank not in ("chain", "size"):
        raise ValueError(f"unknown rank {rank!r}")
    levels = inclusion.rank if rank == "chain" else inclusion.size
    by_rank: dict[int, list[int]] = defaultdict(list)
    for node in projection.nodes:
        by_rank[levels[node.id]].append(node.id)
    order = {
        rank: sorted(members, key=lambda i: projection.nodes[i].label)
        for rank, members in by_rank.items()
    }
    if layout == "sorted":
        return _spread(order)
    neighbors: dict[int, list[int]] = defaultdict(list)
    for a, cov in enumerate(inclusion.covers):
        for b in cov:
            neighbors[a].append(b)
            neighbors[b].append(a)
    ranks = sorted(order)
    for sweep in range(_N_BARYCENTRIC_SWEEPS):
        sweep_ranks = ranks if sweep % 2 == 0 else list(reversed(ranks))
        # The first level in the sweep direction stays fixed as the anchor.
        for level in sweep_ranks[1:]:
            xs = _spread(order)
            key = {
                i: (
                    (
                        sum(xs[j][0] for j in neighbors[i]) / len(neighbors[i])
                        if neighbors[i]
                        else xs[i][0]
                    ),
                    projection.nodes[i].label,
                )
                for i in order[level]
            }
            order[level] = sorted(order[level], key=key.__getitem__)
    return _spread(order)


def _node_sizes(
    projection: PhiStructureProjection, theme: Theme, size_by: str | None
) -> list[float]:
    smin, smax = theme.node_size_range
    if size_by is None:
        return [(smin + smax) / 2.0] * len(projection.nodes)
    return rescale([getattr(n, size_by) for n in projection.nodes], smin, smax)


def render_lattice(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    layout: str = "barycentric",
    order: str = "mechanism",
    rank: str = "chain",
    size_by: str | None = "sum_phi_relations",
    color_by: str = "phi",
) -> go.Figure:
    """Draw an inclusion partial order as a 2-D Hasse diagram."""
    if size_by is not None and size_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown size_by {size_by!r}")
    if color_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown color_by {color_by!r}")
    inclusion = projection.inclusion(order)
    pos = _positions(projection, inclusion, layout=layout, rank=rank)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for a, cov in enumerate(inclusion.covers):
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
            "size": _node_sizes(projection, theme, size_by),
            "color": [getattr(n, color_by) for n in projection.nodes],
            "colorscale": theme.colorscale,
            "colorbar": {"title": CHANNEL_TITLES[color_by]},
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
        yaxis={
            "title": (
                "inclusion rank"
                if rank == "chain"
                else order.replace("_", "-") + " size"
            ),
            "dtick": 1,
        },
    )
    return figure
