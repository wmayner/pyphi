"""Tests for the 3-D simplicial-complex renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def test_geometry_dataclass_frozen():
    import dataclasses

    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    geo = SimplicialComplexGeometry()
    assert geo.max_radius == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        geo.max_radius = 2.0  # type: ignore[misc]


def test_endpoint_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _endpoint_positions

    geo = SimplicialComplexGeometry()
    pos = _endpoint_positions(xor_projection, geo)
    assert set(pos) == set(range(8))
    # Deterministic.
    assert pos == _endpoint_positions(xor_projection, geo)
    # Flat by default.
    assert all(p[2] == 0.0 for p in pos.values())
    # d3's cause/effect share the purview (0,1,2): cause sits -x, effect +x.
    assert pos[6][0] < pos[7][0]
    # Endpoints sharing (purview, direction) are jittered apart:
    # causes of d0, d1, d2, d3 all have purview (0,1,2).
    cause_ids = (0, 2, 4, 6)
    assert len({pos[i] for i in cause_ids}) == 4


def test_mechanism_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
    from pyphi.visualize.render.simplicial_complex import _mechanism_positions

    geo = SimplicialComplexGeometry(max_radius=2.0)
    pos = _mechanism_positions(xor_projection, geo)
    assert set(pos) == {0, 1, 2, 3}
    # Mechanisms are unique, so all positions distinct.
    assert len(set(pos.values())) == 4
    # abc (size 3) sits on the outermost shell at max_radius.
    x, y, _z = pos[3]
    assert (x**2 + y**2) ** 0.5 == pytest.approx(2.0)
