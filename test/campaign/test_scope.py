import pytest

from pyphi import examples
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.campaign.scope import resolve_scope
from pyphi.direction import Direction
from pyphi.serialize import load
from pyphi.serialize import save

CANDIDATES = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def test_unconstrained_admits_everything():
    scope = AxisScope()
    assert scope.unconstrained
    assert list(scope.select(CANDIDATES)) == CANDIDATES


def test_explicit_is_the_axis():
    scope = AxisScope(explicit=((0, 1), (2,)))
    assert list(scope.select(CANDIDATES)) == [(2,), (0, 1)]


def test_explicit_excludes_other_fields():
    with pytest.raises(ValueError, match="exclusive"):
        AxisScope(explicit=((0,),), max_order=2)


def test_constraints_intersect():
    scope = AxisScope(max_order=2, containing=(0,))
    assert list(scope.select(CANDIDATES)) == [(0,), (0, 1), (0, 2)]
    scope = AxisScope(min_order=2, within=(0, 1))
    assert list(scope.select(CANDIDATES)) == [(0, 1)]


def test_ces_scope_directions():
    scope = CESScope(cause_purviews=AxisScope(max_order=1))
    assert scope.purviews(Direction.CAUSE).max_order == 1
    assert scope.purviews(Direction.EFFECT).unconstrained


def test_resolve_scope_coerces_labels():
    substrate = examples.basic_substrate()
    labels = list(map(str, substrate.node_labels))
    scope = CESScope(mechanisms=AxisScope(containing=(labels[0],)))
    resolved = resolve_scope(scope, substrate.node_labels)
    assert resolved.mechanisms.containing == (0,)


def test_scope_roundtrips(tmp_path):
    scope = CESScope(
        mechanisms=AxisScope(explicit=((0, 1), (2,))),
        effect_purviews=AxisScope(max_order=2, within=(0, 1, 2)),
    )
    save(scope, tmp_path / "scope.json.gz")
    assert load(tmp_path / "scope.json.gz") == scope


def test_purview_axis_applies_order_cap():
    scope = CESScope(max_purview_order_by_mechanism_order=((1, 1), (2, 3)))
    axis = scope.purview_axis(Direction.CAUSE, (0,))
    assert axis.admits((0,))
    assert not axis.admits((0, 1))
    axis2 = scope.purview_axis(Direction.EFFECT, (0, 1))
    assert axis2.admits((0, 1, 2))
    assert not axis2.admits((0, 1, 2, 3))


def test_purview_axis_falls_back_for_unlisted_orders():
    scope = CESScope(
        cause_purviews=AxisScope(max_order=2),
        max_purview_order_by_mechanism_order=((1, 1),),
    )
    # order-3 mechanism is not in the table: static cap alone applies
    axis = scope.purview_axis(Direction.CAUSE, (0, 1, 2))
    assert axis.max_order == 2


def test_purview_axis_intersects_with_static_cap():
    scope = CESScope(
        cause_purviews=AxisScope(max_order=2),
        max_purview_order_by_mechanism_order=((1, 5),),
    )
    # the static cap is tighter than the table's: intersection wins
    assert scope.purview_axis(Direction.CAUSE, (0,)).max_order == 2


def test_purview_axis_filters_explicit_lists():
    scope = CESScope(
        cause_purviews=AxisScope(explicit=((0,), (0, 1))),
        max_purview_order_by_mechanism_order=((1, 1),),
    )
    axis = scope.purview_axis(Direction.CAUSE, (0,))
    assert axis.explicit == ((0,),)


def test_order_cap_table_validation():
    with pytest.raises(ValueError, match="unique"):
        CESScope(max_purview_order_by_mechanism_order=((1, 1), (1, 2)))
    with pytest.raises(ValueError, match="positive"):
        CESScope(max_purview_order_by_mechanism_order=((0, 1),))
    with pytest.raises(ValueError, match="positive"):
        CESScope(max_purview_order_by_mechanism_order=((1, 0),))


def test_order_cap_survives_resolution_and_serialization(tmp_path):
    substrate = examples.basic_substrate()
    scope = CESScope(max_purview_order_by_mechanism_order=((1, 2),))
    resolved = resolve_scope(scope, substrate.node_labels)
    assert resolved.max_purview_order_by_mechanism_order == ((1, 2),)
    path = tmp_path / "scope.json.gz"
    save(resolved, path)
    assert load(path).max_purview_order_by_mechanism_order == ((1, 2),)
