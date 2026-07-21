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
