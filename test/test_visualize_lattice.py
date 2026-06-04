"""Figure-structure tests for the lattice renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def test_lattice_figure_structure(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(xor_projection, DEFAULT_THEME)
    # Two traces: edges (lines) then nodes (markers).
    assert len(fig.data) == 2
    edge_trace, node_trace = fig.data
    n = len(xor_projection.nodes)
    n_edges = sum(len(c) for c in xor_projection.inclusion.covers)
    # Edge trace: each cover edge contributes (x0, x1, None).
    assert len(edge_trace.x) == 3 * n_edges
    # Node trace: one marker per distinction, y = inclusion rank.
    assert len(node_trace.x) == n
    assert tuple(node_trace.y) == tuple(float(r) for r in xor_projection.inclusion.rank)
    # Hover text mentions each distinction's label.
    for node, text in zip(xor_projection.nodes, node_trace.hovertext, strict=True):
        assert node.label in text


def test_plot_phi_structure_lattice_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    fig = plot_phi_structure(ces, view="lattice")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_phi_structure_unimplemented_views_raise():
    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    for view in ("evocative", "scatter", "matrix"):
        with pytest.raises(NotImplementedError, match=view):
            plot_phi_structure(ces, view=view)
