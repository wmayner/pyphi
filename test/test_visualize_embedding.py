import numpy as np


def test_mice_vectors_blocks_set():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import _MECHANISM_WEIGHT
    from pyphi.visualize.render.embedding import _PURVIEW_WEIGHT
    from pyphi.visualize.render.embedding import _mice_vectors

    # One distinction (mechanism (0,)), cause purview (1, 2), effect purview (2,).
    nodes = [
        SimpleNamespace(id=0, mechanism=(0,), cause_purview=(1, 2), effect_purview=(2,))
    ]
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(1, 2)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    v = _mice_vectors(proj)
    # Units present: {0, 1, 2}; width = 3; columns 0..2 purview, 3..5 mechanism,
    # 6 direction.
    assert v.shape == (2, 7)
    # Cause row: purview {1,2} -> cols 1,2; mechanism {0} -> col 3; direction -.
    assert v[0, 1] == _PURVIEW_WEIGHT and v[0, 2] == _PURVIEW_WEIGHT
    assert v[0, 0] == 0.0
    assert v[0, 3] == _MECHANISM_WEIGHT
    assert v[0, 6] < 0  # cause marker negative
    assert v[1, 6] > 0  # effect marker positive


def test_mice_vectors_honors_custom_weights():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import _mice_vectors

    nodes = [
        SimpleNamespace(id=0, mechanism=(0,), cause_purview=(1, 2), effect_purview=(2,))
    ]
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(1, 2)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    v = _mice_vectors(
        proj, purview_weight=2.0, mechanism_weight=3.0, direction_weight=4.0
    )
    # Purview block (cols 1, 2), mechanism block (col 3), direction marker (col 6).
    assert v[0, 1] == 2.0 and v[0, 2] == 2.0
    assert v[0, 3] == 3.0
    assert v[0, 6] == -4.0 and v[1, 6] == 4.0


def test_mice_distance_honors_custom_weights():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import _mice_distance

    nodes = [
        SimpleNamespace(
            id=0, mechanism=(0,), cause_purview=(0, 1), effect_purview=(0, 1)
        ),
        SimpleNamespace(id=1, mechanism=(1,), cause_purview=(0, 1), effect_purview=(2,)),
    ]
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(0, 1)),
        SimpleNamespace(id=2, distinction_id=1, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=3, distinction_id=1, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    # Zeroing the mechanism and direction weights leaves pure purview Jaccard:
    # ep0/ep2 share purview (0,1) -> 0; ep0/ep3 are disjoint -> 1.
    d = _mice_distance(
        proj, purview_weight=1.0, mechanism_weight=0.0, direction_weight=0.0
    )
    assert d[0, 2] == 0.0
    assert d[0, 3] == 1.0


def test_mice_distance_purview_overlap_orders_pairs():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import _mice_distance

    nodes = [
        SimpleNamespace(
            id=0, mechanism=(0,), cause_purview=(0, 1), effect_purview=(0, 1)
        ),
        SimpleNamespace(id=1, mechanism=(1,), cause_purview=(0, 1), effect_purview=(2,)),
    ]
    # ep0,ep2 share purview (0,1); ep0,ep3 are disjoint ((0,1) vs (2,)).
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(0, 1)),
        SimpleNamespace(id=2, distinction_id=1, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=3, distinction_id=1, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    d = _mice_distance(proj)
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)
    # Sharing a purview is nearer than disjoint purviews.
    assert d[0, 2] < d[0, 3]


def test_pca_embed_shape_centered_deterministic():
    from pyphi.visualize.render.embedding import pca_embed

    rng = np.arange(30, dtype=float).reshape(10, 3)
    a = pca_embed(rng)
    assert a.shape == (10, 3)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-9)  # centered
    assert np.array_equal(a, pca_embed(rng))  # deterministic


def test_pca_embed_degenerate_fallback():
    from pyphi.visualize.render.embedding import pca_embed

    # All rows identical: zero variance -> every component falls back to a
    # spread by id, so points are still distinct.
    flat = np.ones((4, 5))
    a = pca_embed(flat)
    assert not np.any(np.isnan(a))
    assert len({tuple(row) for row in a}) == 4


def test_mds_embed_recovers_line_order():
    from pyphi.visualize.render.embedding import mds_embed

    # Four points on a line at 0,1,2,3: classical MDS recovers their order on
    # the first axis (up to sign, which the sign-fix pins).
    pts = np.array([0.0, 1.0, 2.0, 3.0])
    dist = np.abs(pts[:, None] - pts[None, :])
    a = mds_embed(dist)
    assert a.shape == (4, 3)
    first = a[:, 0]
    order = np.argsort(first)
    assert list(order) == [0, 1, 2, 3] or list(order) == [3, 2, 1, 0]


def test_mds_embed_deterministic():
    from pyphi.visualize.render.embedding import mds_embed

    dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    assert np.array_equal(mds_embed(dist), mds_embed(dist))


def test_embedding_positions_threads_geometry_weights():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import embedding_positions
    from pyphi.visualize.render.hypergraph import HypergraphGeometry

    # Two distinctions sharing every purview: only the mechanism and direction
    # blocks can separate their endpoints, so the geometry weights must reach
    # the distance builder for the layout to respond.
    nodes = [
        SimpleNamespace(
            id=0, mechanism=(0,), cause_purview=(0, 1), effect_purview=(0, 1)
        ),
        SimpleNamespace(
            id=1, mechanism=(1,), cause_purview=(0, 1), effect_purview=(0, 1)
        ),
    ]
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(0, 1)),
        SimpleNamespace(id=2, distinction_id=1, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=3, distinction_id=1, direction="effect", purview=(0, 1)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    with_mech, _ = embedding_positions(
        proj, HypergraphGeometry(embed_mechanism_weight=2.0)
    )
    without_mech, _ = embedding_positions(
        proj,
        HypergraphGeometry(embed_mechanism_weight=0.0, embed_direction_weight=0.0),
    )
    assert with_mech != without_mech
