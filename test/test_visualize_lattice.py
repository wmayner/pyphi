"""Figure-structure tests for the lattice renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


@pytest.fixture
def crossing_projection():
    """Two ranks whose cover edges cross under label-sorted placement.

    Rank 0 holds ``a`` (id 0) and ``b`` (id 1); rank 1 holds ``c`` (id 2,
    covering ``b``) and ``d`` (id 3, covering ``a``). Label order puts ``c``
    left of ``d``, so the edges c-b and d-a cross; barycentric ordering
    swaps them.
    """
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import DistinctionNode
    from pyphi.visualize.projection import InclusionOrder
    from pyphi.visualize.projection import PhiStructureProjection

    def node(i, label):
        return DistinctionNode(
            id=i,
            mechanism=(i,),
            label=label,
            cause_purview=(i,),
            effect_purview=(i,),
            mechanism_state=(0,),
            phi=1.0,
            sum_phi_relations=0.0,
            includes=False,
            included=False,
        )

    return PhiStructureProjection(
        nodes=(node(0, "a"), node(1, "b"), node(2, "c"), node(3, "d")),
        edges=(),
        inclusion=InclusionOrder(covers=((), (), (1,), (0,)), rank=(0, 0, 1, 1)),
        node_labels=NodeLabels(("A", "B", "C", "D"), (0, 1, 2, 3)),
    )


def test_sorted_layout_orders_by_label(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    pos = _positions(crossing_projection, layout="sorted")
    assert pos[2][0] < pos[3][0]


def test_barycentric_layout_uncrosses_edges(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    pos = _positions(crossing_projection, layout="barycentric")
    # d (covering a, on the left) moves left of c (covering b, on the right).
    assert pos[3][0] < pos[2][0]
    assert pos[0][0] < pos[1][0]
    # y is still the inclusion rank.
    assert pos[0][1] == pos[1][1] == 0.0
    assert pos[2][1] == pos[3][1] == 1.0


def test_unknown_layout_raises(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    with pytest.raises(ValueError, match="layout"):
        _positions(crossing_projection, layout="bogus")


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
    for layout in ("barycentric", "sorted"):
        fig = plot_phi_structure(ces, view="lattice", layout=layout)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


def test_plot_phi_structure_unimplemented_views_raise():
    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    for view in ("evocative", "scatter", "matrix"):
        with pytest.raises(NotImplementedError, match=view):
            plot_phi_structure(ces, view=view)
