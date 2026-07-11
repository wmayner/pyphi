"""phi-by-degree relation spectrum panel."""

from __future__ import annotations

import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.theme import Theme


def render_relation_spectrum(
    projection: CESProjection, theme: Theme, fig: go.Figure | None = None
) -> go.Figure:
    """A 2-D bar panel of relation count and sum of phi per relation degree.

    Reads the projection's closed-form ``degree_spectrum`` (count and Σφ_r per
    degree), so the high-degree structure that is hard to read in the 3-D
    hypergraph view is summarized exactly, whatever relation cap the other
    views use.
    """
    spectrum = projection.degree_spectrum
    degrees = sorted(spectrum)
    count = {d: spectrum[d][0] for d in degrees}
    sum_phi = {d: spectrum[d][1] for d in degrees}
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
