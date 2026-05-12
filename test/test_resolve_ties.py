from pyphi import config
from pyphi import resolve_ties
from pyphi.direction import Direction
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import EdgeCut
from pyphi.models.partitions import NullCut


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


def test_lex_key_nullcut_is_empty_bytes():
    null = NullCut(indices=(0, 1, 2))
    assert null.lex_key() == b""


def test_lex_key_directed_bipartition_matches_cut_matrix_bytes():
    sp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    expected = sp.cut_matrix(3).tobytes()
    assert sp.lex_key() == expected


def test_lex_key_equivalent_partitions_compare_equal():
    """Two partitions inducing the same edge cut compare equal under lex_key.

    A 3-node EFFECT bipartition severing 1→2 has cut_matrix
    [[0,0,0],[0,0,1],[0,0,0]]; an EdgeCut with the same matrix on the
    same node set must produce the same lex_key.
    """
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    matrix = bp.cut_matrix(3)
    ec = EdgeCut(node_indices=(0, 1, 2), cut_matrix=matrix)
    assert bp.lex_key() == ec.lex_key()


def test_lex_key_distinct_partitions_compare_distinct():
    sp_a = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sp_b = DirectedBipartition(Direction.EFFECT, (0,), (2,))
    assert sp_a.lex_key() != sp_b.lex_key()


def test_lex_key_is_total_ordering():
    """Distinct partitions sort under bytes comparison."""
    sp_a = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sp_b = DirectedBipartition(Direction.EFFECT, (0,), (2,))
    assert sp_a.lex_key() < sp_b.lex_key() or sp_b.lex_key() < sp_a.lex_key()


class DummySia:
    """Minimal SIA-shaped object for resolve_ties.sias tests."""

    def __init__(self, normalized_phi, phi, partition):
        self.normalized_phi = normalized_phi
        self.phi = phi
        self.partition = partition


def test_partition_lex_strategy_returns_partition_bytes():
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sia = DummySia(normalized_phi=0.0, phi=0.0, partition=bp)
    key_fn = resolve_ties.phi_object_tie_resolution_strategies["PARTITION_LEX"]
    assert key_fn(sia) == bp.lex_key()


def test_sias_resolves_partition_lex_tertiary_tiebreak():
    """When (normalized_phi, -phi) ties, PARTITION_LEX picks the smallest lex key."""
    bp_hi = DirectedBipartition(
        Direction.EFFECT, (0,), (2,)
    )  # larger lex_key (0→2 at index 2)
    bp_lo = DirectedBipartition(
        Direction.EFFECT, (1,), (2,)
    )  # smaller lex_key (1→2 at index 5)
    assert bp_lo.lex_key() < bp_hi.lex_key()
    a = DummySia(normalized_phi=0.5, phi=1.0, partition=bp_lo)
    b = DummySia(normalized_phi=0.5, phi=1.0, partition=bp_hi)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([b, a]))  # b first to exercise ordering
    assert resolved == [a]


def test_sias_falls_through_to_secondary_when_normalized_phi_differs():
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    smaller = DummySia(normalized_phi=0.3, phi=1.0, partition=bp)
    larger = DummySia(normalized_phi=0.5, phi=1.0, partition=bp)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([smaller, larger]))
    assert resolved == [smaller]


def test_sias_secondary_prefers_larger_unnormalized_phi():
    """At equal normalized_phi, NEGATIVE_PHI minimised picks larger phi."""
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    lower_phi = DummySia(normalized_phi=0.5, phi=1.0, partition=bp)
    higher_phi = DummySia(normalized_phi=0.5, phi=2.0, partition=bp)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([lower_phi, higher_phi]))
    assert resolved == [higher_phi]
