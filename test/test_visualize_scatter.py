"""Tests for the relational-role scatter renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def _make_projection(nodes, edges=()):
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import InclusionOrder
    from pyphi.visualize.projection import PhiStructureProjection

    n = len(nodes)
    order = InclusionOrder(covers=((),) * n, rank=(0,) * n, size=(1,) * n)
    return PhiStructureProjection(
        nodes=tuple(nodes),
        edges=tuple(edges),
        mechanism_inclusion=order,
        purview_union_inclusion=order,
        node_labels=NodeLabels(("A", "B", "C", "D"), (0, 1, 2, 3)),
    )


def _node(i, label, purview, phi=1.0, sum_phi=0.0, includes=False, included=False):
    from pyphi.visualize.projection import DistinctionNode

    return DistinctionNode(
        id=i,
        mechanism=(i,),
        label=label,
        cause_purview=purview,
        effect_purview=purview,
        mechanism_state=(0,),
        phi=phi,
        sum_phi_relations=sum_phi,
        includes=includes,
        included=included,
    )


@pytest.fixture
def varied_projection():
    """Distinct singleton purview unions (non-degenerate PCA), varied roles,
    one node disconnected."""
    from pyphi.visualize.projection import RelationEdge

    nodes = [
        _node(0, "a", (0,), sum_phi=1.0, includes=True, included=True),
        _node(1, "b", (1,), sum_phi=2.0, includes=True),
        _node(2, "c", (2,), sum_phi=3.0, included=True),
        _node(3, "d", (3,), sum_phi=0.0),
    ]
    edges = (
        RelationEdge(relata=(0, 1), degree=2, phi=1.0, overlap=()),
        RelationEdge(relata=(0, 1, 2), degree=3, phi=0.5, overlap=()),
        # A self-relation does not make node 3 "connected".
        RelationEdge(relata=(3,), degree=1, phi=0.2, overlap=()),
    )
    return _make_projection(nodes, edges)


def _render(projection, **kwargs):
    from pyphi.visualize.render.scatter import render_scatter
    from pyphi.visualize.theme import DEFAULT_THEME

    return render_scatter(projection, DEFAULT_THEME, **kwargs)


def test_scatter_figure_structure(varied_projection):
    import plotly.graph_objects as go

    fig = _render(varied_projection)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert len(trace.x) == 4
    assert tuple(trace.text) == ("a", "b", "c", "d")
    # Largest marker belongs to the highest sum_phi_relations.
    sizes = list(trace.marker.size)
    assert sizes.index(max(sizes)) == 2
    # Role colors from the theme.
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert tuple(trace.marker.color) == (
        roles["extended"],
        roles["includes"],
        roles["included"],
        roles["none"],
    )
    # Connectedness symbols: node 3 only self-relates.
    assert tuple(trace.marker.symbol) == (
        "circle",
        "circle",
        "circle",
        "diamond-open",
    )


def test_scatter_positions_deterministic_and_distinct(varied_projection):
    a = _render(varied_projection).data[0]
    b = _render(varied_projection).data[0]
    assert tuple(a.x) == tuple(b.x) and tuple(a.y) == tuple(b.y)
    coords = set(zip(a.x, a.y, strict=True))
    assert len(coords) == 4


def test_scatter_degenerate_fallback(xor_projection):
    # All xor purview unions are identical: PCA variance is zero, the
    # fallback spreads points by node id.
    trace = _render(xor_projection).data[0]
    coords = set(zip(trace.x, trace.y, strict=True))
    assert len(coords) == 4
    # All roles are "none" and everything is connected.
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert set(trace.marker.color) == {roles["none"]}
    assert set(trace.marker.symbol) == {"circle"}


def test_scatter_numeric_color_channel(varied_projection):
    trace = _render(varied_projection, color_by="phi").data[0]
    assert tuple(trace.marker.color) == (1.0, 1.0, 1.0, 1.0)


def test_scatter_invalid_channels_raise(varied_projection):
    with pytest.raises(ValueError, match="size_by"):
        _render(varied_projection, size_by="bogus")
    with pytest.raises(ValueError, match="color_by"):
        _render(varied_projection, color_by="bogus")
