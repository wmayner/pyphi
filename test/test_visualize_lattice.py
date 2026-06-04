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

    order = InclusionOrder(
        covers=((), (), (1,), (0,)), rank=(0, 0, 1, 1), size=(1, 1, 2, 2)
    )
    return PhiStructureProjection(
        nodes=(node(0, "a"), node(1, "b"), node(2, "c"), node(3, "d")),
        edges=(),
        mechanism_inclusion=order,
        purview_union_inclusion=order,
        node_labels=NodeLabels(("A", "B", "C", "D"), (0, 1, 2, 3)),
    )


def test_sorted_layout_orders_by_label(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    pos = _positions(
        crossing_projection, crossing_projection.mechanism_inclusion, layout="sorted"
    )
    assert pos[2][0] < pos[3][0]


def test_barycentric_layout_uncrosses_edges(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    pos = _positions(
        crossing_projection,
        crossing_projection.mechanism_inclusion,
        layout="barycentric",
    )
    # d (covering a, on the left) moves left of c (covering b, on the right).
    assert pos[3][0] < pos[2][0]
    assert pos[0][0] < pos[1][0]
    # y is still the inclusion rank.
    assert pos[0][1] == pos[1][1] == 0.0
    assert pos[2][1] == pos[3][1] == 1.0


def test_unknown_layout_raises(crossing_projection):
    from pyphi.visualize.render.lattice import _positions

    with pytest.raises(ValueError, match="layout"):
        _positions(
            crossing_projection, crossing_projection.mechanism_inclusion, layout="bogus"
        )


def test_lattice_figure_structure(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(xor_projection, DEFAULT_THEME)
    # Two traces: edges (lines) then nodes (markers).
    assert len(fig.data) == 2
    edge_trace, node_trace = fig.data
    n = len(xor_projection.nodes)
    n_edges = sum(len(c) for c in xor_projection.mechanism_inclusion.covers)
    # Edge trace: each cover edge contributes (x0, x1, None).
    assert len(edge_trace.x) == 3 * n_edges
    # Node trace: one marker per distinction, y = inclusion rank.
    assert len(node_trace.x) == n
    assert tuple(node_trace.y) == tuple(
        float(r) for r in xor_projection.mechanism_inclusion.rank
    )
    # Hover text mentions each distinction's label.
    for node, text in zip(xor_projection.nodes, node_trace.hovertext, strict=True):
        assert node.label in text


def test_plot_phi_structure_lattice_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    for layout in ("barycentric", "sorted"):
        for order in ("mechanism", "purview_union"):
            fig = plot_phi_structure(ces, view="lattice", layout=layout, order=order)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) == 2


def test_lattice_rank_size_uses_cardinality(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(xor_projection, DEFAULT_THEME, rank="size")
    node_trace = fig.data[1]
    # Mechanism sizes for ab, ac, bc, abc.
    assert tuple(node_trace.y) == (2.0, 2.0, 2.0, 3.0)


def test_lattice_size_by_and_color_by(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(
        xor_projection, DEFAULT_THEME, size_by="phi", color_by="sum_phi_relations"
    )
    node_trace = fig.data[1]
    phis = [n.phi for n in xor_projection.nodes]
    sizes = list(node_trace.marker.size)
    # The largest marker belongs to the highest-phi node.
    assert sizes.index(max(sizes)) == phis.index(max(phis))
    assert tuple(node_trace.marker.color) == tuple(
        n.sum_phi_relations for n in xor_projection.nodes
    )

    fig = render_lattice(xor_projection, DEFAULT_THEME, size_by=None)
    node_trace = fig.data[1]
    assert len(set(node_trace.marker.size)) == 1


def test_lattice_invalid_channels_raise(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    with pytest.raises(ValueError, match="rank"):
        render_lattice(xor_projection, DEFAULT_THEME, rank="bogus")
    with pytest.raises(ValueError, match="size_by"):
        render_lattice(xor_projection, DEFAULT_THEME, size_by="bogus")
    with pytest.raises(ValueError, match="color_by"):
        render_lattice(xor_projection, DEFAULT_THEME, color_by="bogus")


def test_lattice_order_selects_partial_order(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(xor_projection, DEFAULT_THEME, order="purview_union")
    node_trace = fig.data[1]
    assert tuple(node_trace.y) == tuple(
        float(r) for r in xor_projection.purview_union_inclusion.rank
    )


def test_plot_phi_structure_unknown_view_raises():
    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    with pytest.raises(ValueError, match="view"):
        plot_phi_structure(ces, view="bogus")
