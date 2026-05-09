from pyphi import config
from pyphi import resolve_ties


class DummyPhiObject:
    def __init__(self, phi, purview, normalized_phi=None):
        self.phi = phi
        self.purview = purview
        self.normalized_phi = phi if normalized_phi is None else normalized_phi


def test_resolve_none_returns_objects():
    objects = [DummyPhiObject(1.0, (0,)), DummyPhiObject(2.0, (0, 1))]
    resolved = list(resolve_ties.resolve(objects, "NONE", operation=max))
    assert resolved == objects


def test_resolve_max_phi_ties():
    low = DummyPhiObject(1.0, (0,))
    high_a = DummyPhiObject(2.0, (0, 1))
    high_b = DummyPhiObject(2.0, (1, 2))
    resolved = list(resolve_ties.resolve([low, high_a, high_b], "PHI", operation=max))
    assert resolved == [high_a, high_b]


def test_resolve_multiple_strategies_selects_smallest_purview():
    larger = DummyPhiObject(1.0, (0, 1))
    smaller = DummyPhiObject(1.0, (0,))
    resolved = list(
        resolve_ties.resolve(
            [larger, smaller],
            ["PHI", "NEGATIVE_PURVIEW_SIZE"],
            operation=max,
        )
    )
    assert resolved == [smaller]


def test_states_uses_config_default():
    low = DummyPhiObject(1.0, (0,))
    high = DummyPhiObject(2.0, (0, 1))
    with config.override(state_tie_resolution="PHI"):
        resolved = list(resolve_ties.states([low, high]))
    assert resolved == [high]


def test_partitions_uses_min_operation():
    low = DummyPhiObject(1.0, (0,))
    high = DummyPhiObject(2.0, (0, 1))
    with config.override(mip_tie_resolution="PHI"):
        resolved = list(resolve_ties.partitions([low, high]))
    assert resolved == [low]


def test_purviews_uses_max_operation():
    low = DummyPhiObject(1.0, (0,))
    high = DummyPhiObject(2.0, (0, 1))
    with config.override(purview_tie_resolution="PHI"):
        resolved = list(resolve_ties.purviews([low, high]))
    assert resolved == [high]
