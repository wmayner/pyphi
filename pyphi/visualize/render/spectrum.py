"""phi-by-degree relation spectrum panel."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.theme import Theme


def render_relation_spectrum(
    projection: CESProjection, theme: Theme, fig: go.Figure | None = None
) -> go.Figure:
    """A 2-D bar panel of relation count and sum of phi per relation degree.

    Computed from the projection's relation-level ``edges`` (which carry every
    relation's degree and phi), so the high-degree structure that is hard to
    read in the 3-D simplicial-complex view is summarized at a glance.
    """
    count: dict[int, int] = defaultdict(int)
    sum_phi: dict[int, float] = defaultdict(float)
    for edge in projection.edges:
        count[edge.degree] += 1
        sum_phi[edge.degree] += edge.phi
    degrees = sorted(count)
    figure = go.Figure() if fig is None else fig
    figure.add_trace(
        go.Bar(
            x=degrees,
            y=[sum_phi[d] for d in degrees],
            customdata=[[count[d]] for d in degrees],
            marker={
                "color": [sum_phi[d] for d in degrees],
                "colorscale": theme.face_colorscale,
            },
            hovertemplate=(
                "degree %{x}<br>Σφ = %{y:.4g}<br>count = %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis={"title": "relation degree", "dtick": 1},
        yaxis={"title": "Σφ"},
        paper_bgcolor=theme.background,
        font={"family": theme.font_family},
        showlegend=False,
    )
    return figure
