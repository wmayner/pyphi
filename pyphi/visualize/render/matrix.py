"""Relation-matrix renderer for CES projections."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.theme import Theme


def render_matrix(
    projection: CESProjection,
    theme: Theme,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Heatmap of relation strength between pairs of distinctions.

    An off-diagonal cell sums the phi of every relation involving both
    distinctions; a diagonal cell sums the distinction's self-relations
    (its reflexivity). Rows and columns are ordered by mechanism size,
    then label, so mechanism orders form contiguous blocks.
    """
    order = sorted(projection.nodes, key=lambda n: (len(n.mechanism), n.label))
    pos = {n.id: k for k, n in enumerate(order)}
    labels = [n.label for n in order]
    n = len(order)
    z = np.zeros((n, n))
    for e in projection.edges:
        relata = set(e.relata)
        if len(relata) == 1:
            (i,) = relata
            z[pos[i], pos[i]] += e.phi
        else:
            for a, b in combinations(sorted(relata), 2):
                z[pos[a], pos[b]] += e.phi
                z[pos[b], pos[a]] += e.phi
    hover = [
        [f"{labels[r]} · {labels[c]}<br>Σφ = {z[r, c]:.4g}" for c in range(n)]
        for r in range(n)
    ]
    trace = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=theme.colorscale,
        colorbar={"title": "Σφ"},
        hovertext=hover,
        hoverinfo="text",
    )
    figure = go.Figure() if fig is None else fig
    figure.add_trace(trace)
    figure.update_layout(
        plot_bgcolor=theme.background,
        font={"family": theme.font_family},
        yaxis={"autorange": "reversed"},
    )
    return figure
