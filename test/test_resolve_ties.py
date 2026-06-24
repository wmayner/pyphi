import numpy as np
import pytest

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


def test_lex_key_directed_bipartition_encodes_severed_edge():
    """A 3-node EFFECT bipartition severing 1→2 has a 1 at row 1 col 2."""
    sp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    expected = np.zeros((3, 3), dtype=np.uint8)
    expected[1, 2] = 1
    assert sp.lex_key() == expected.tobytes()


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


def test_lex_key_complete_edge_cut_is_all_ones():
    """CompleteEdgeCut severs every connection — key is an all-ones matrix."""
    from pyphi.models.partitions import CompleteEdgeCut

    cec = CompleteEdgeCut(node_indices=(0, 1, 2))
    expected = np.ones((3, 3), dtype=np.uint8).tobytes()
    assert cec.lex_key() == expected


def test_lex_key_directed_set_partition_matches_induced_cut():
    """DirectedSetPartition lex_key encodes the same cut as its cut_matrix."""
    from pyphi.models.partitions import DirectedSetPartition

    # Build a 3-node partition with parts [[0], [1, 2]]; sever 0→1 and 0→2.
    node_indices = (0, 1, 2)
    cut_mat = np.zeros((3, 3), dtype=int)
    cut_mat[0, 1] = 1
    cut_mat[0, 2] = 1
    sp = DirectedSetPartition(
        node_indices=node_indices,
        cut_matrix=cut_mat,
        set_partition=[[0], [1, 2]],
    )
    expected = sp.cut_matrix(3).astype(np.uint8)
    assert sp.lex_key() == expected.tobytes()


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


class DummyClampedMip:
    """MIP-shaped object with both clamped ``phi`` and raw ``signed_phi``.

    Mirrors the post-clamp shape of ``RepertoireIrreducibilityAnalysis``
    and ``SystemIrreducibilityAnalysis`` for partitions whose raw
    integration value is non-positive.
    """

    def __init__(self, phi, normalized_phi, signed_phi, partition):
        self.phi = phi
        self.normalized_phi = normalized_phi
        self.signed_phi = signed_phi
        self.partition = partition


def test_default_mip_tie_resolution_does_not_consult_signed_phi():
    """Two MIP candidates clamped to phi=0 with differing signed_phi tie.

    The ``|·|+`` clamp (Eqs 19-20) maps any non-positive integration
    value to zero. The default ``mip_tie_resolution`` chain keys on the
    clamped ``phi`` only, so partitions whose raw ``signed_phi`` differs
    but whose clamped ``phi`` matches surface as a tied set; their
    signed value is preserved as diagnostic metadata, not as a
    tie-break key.
    """
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    less_negative = DummyClampedMip(
        phi=0.0, normalized_phi=0.0, signed_phi=-0.5, partition=bp
    )
    more_negative = DummyClampedMip(
        phi=0.0, normalized_phi=0.0, signed_phi=-1.0, partition=bp
    )
    resolved = list(resolve_ties.partitions([less_negative, more_negative]))
    assert resolved == [less_negative, more_negative]


def test_default_mip_tie_chain_does_not_register_signed_phi_strategy():
    """``signed_phi`` has no registered phi-object tie-resolution strategy.

    The Q3 design decision is that signed_phi is diagnostic-only: no
    strategy keys on it, and the default tie chain falls through to
    deterministic lex-canonical resolution at higher cascade levels
    rather than preferring a less-negative raw integration.
    """
    assert "NEGATIVE_SIGNED_PHI" not in resolve_ties.phi_object_tie_resolution_strategies
    assert "SIGNED_PHI" not in resolve_ties.phi_object_tie_resolution_strategies
    default_chain = config.formalism.iit.mip_tie_resolution
    assert all("SIGNED" not in s for s in default_chain)


def test_sia_tie_chain_with_clamped_phi_falls_through_to_partition_lex():
    """Default ``sia_tie_resolution`` resolves clamped-phi ties via lex.

    At equal ``normalized_phi`` and equal clamped ``phi``, the default
    ``sia_tie_resolution = [NORMALIZED_PHI, NEGATIVE_PHI, PARTITION_LEX]``
    breaks the tie on canonical partition order, regardless of any
    difference in ``signed_phi``.
    """
    bp_hi = DirectedBipartition(Direction.EFFECT, (0,), (2,))
    bp_lo = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    assert bp_lo.lex_key() < bp_hi.lex_key()
    less_negative_hi_lex = DummyClampedMip(
        phi=0.0, normalized_phi=0.0, signed_phi=-0.5, partition=bp_hi
    )
    more_negative_lo_lex = DummyClampedMip(
        phi=0.0, normalized_phi=0.0, signed_phi=-1.0, partition=bp_lo
    )
    resolved = list(resolve_ties.sias([less_negative_hi_lex, more_negative_lo_lex]))
    assert resolved == [more_negative_lo_lex]


class DummyIIT3Sia:
    """Minimal IIT 3.0-SIA-shaped object for resolve_iit3_complex_tie tests."""

    def __init__(self, node_indices):
        self.node_indices = node_indices


def test_resolve_iit3_complex_tie_single_candidate_resolves():
    """A single SIA candidate resolves trivially."""
    sia = DummyIIT3Sia(node_indices=(0, 1, 2))
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_iit3_complex_tie([sia], context=ctx)
    assert outcome.outcome == "RESOLVED"
    assert outcome.resolved is sia
    assert outcome.tied_set == (sia,)


def test_resolve_iit3_complex_tie_multi_candidate_returns_unresolved():
    """Multiple candidates flag UNRESOLVED_WITHIN_BUDGET with a lex-smallest representative.

    IIT 3.0 (Oizumi et al. 2014) provides no system-level tie-break
    postulate; the cascade picks a representative for diagnostic
    display but signals indeterminacy so the caller treats the
    clique as failing the exclusion postulate.
    """
    sia_lo = DummyIIT3Sia(node_indices=(0, 2))
    sia_hi = DummyIIT3Sia(node_indices=(1, 2))
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_iit3_complex_tie([sia_hi, sia_lo], context=ctx)
    assert outcome.outcome == "UNRESOLVED_WITHIN_BUDGET"
    assert outcome.resolved is sia_lo  # lex-smallest representative
    assert set(outcome.tied_set) == {sia_lo, sia_hi}
    assert outcome.cascade_level == "Exclusion"


def test_resolve_iit3_complex_tie_empty_raises():
    """Empty candidate list is a programming error."""
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    with pytest.raises(ValueError, match="at least one"):
        resolve_ties.resolve_iit3_complex_tie([], context=ctx)


def test_resolve_iit3_complex_tie_representative_lex_canonical():
    """Representative is computed via tuple(sorted(node_indices))."""
    # (1, 0) sorts to (0, 1); (0, 2) is already sorted. (0, 1) < (0, 2).
    sia_a = DummyIIT3Sia(node_indices=(1, 0))
    sia_b = DummyIIT3Sia(node_indices=(0, 2))
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_iit3_complex_tie([sia_b, sia_a], context=ctx)
    assert outcome.resolved is sia_a


def test_resolve_iit3_complex_tie_on_unresolved_fail_raises_not_a_complex():
    """on_unresolved='fail' raises NotAComplex carrying the tied set."""
    sia_a = DummyIIT3Sia(node_indices=(0, 2))
    sia_b = DummyIIT3Sia(node_indices=(1, 2))
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    with pytest.raises(resolve_ties.NotAComplex) as exc_info:
        resolve_ties.resolve_iit3_complex_tie(
            [sia_a, sia_b], context=ctx, on_unresolved="fail"
        )
    assert set(exc_info.value.tied_set) == {sia_a, sia_b}
    assert exc_info.value.cascade_level == "Exclusion"


def test_resolve_iit3_complex_tie_on_unresolved_warn_emits_warning():
    """on_unresolved='warn' returns UNRESOLVED_WITHIN_BUDGET and emits a UserWarning."""
    sia_a = DummyIIT3Sia(node_indices=(0, 2))
    sia_b = DummyIIT3Sia(node_indices=(1, 2))
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    with pytest.warns(UserWarning, match="cross-subsystem complex tie"):
        outcome = resolve_ties.resolve_iit3_complex_tie(
            [sia_a, sia_b], context=ctx, on_unresolved="warn"
        )
    assert outcome.outcome == "UNRESOLVED_WITHIN_BUDGET"


def test_iit3_sia_map_reduce_consults_sia_tie_resolution():
    """The IIT 3.0 within-subsystem MIP selection routes through
    resolve_ties.sias and consults config.formalism.iit.sia_tie_resolution.

    Override the strategy to a registered-but-bogus name; the call must
    raise KeyError. The default preset value is ["PHI", "PARTITION_LEX"]
    which selects argmin raw phi (paper-canonical IIT 3.0 within-subsystem
    MIP) with lex-canonical partition tie-break.
    """
    from dataclasses import replace

    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.system import System

    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    bad = {**presets.iit3}
    bad["iit"] = replace(bad["iit"], sia_tie_resolution=["DEFINITELY_NOT_A_STRATEGY"])
    with config.override(**bad):
        sys = System.from_substrate(substrate, state, substrate.node_indices)
        with pytest.raises(KeyError):
            sys.sia()


def test_iit3_default_sia_tie_resolution_is_phi_partition_lex():
    """presets.iit3's sia_tie_resolution is ["PHI", "PARTITION_LEX"]."""
    from pyphi.conf import presets

    assert presets.iit3["iit"].sia_tie_resolution == ["PHI", "PARTITION_LEX"]


def test_iit3_find_mip_consults_mip_tie_resolution():
    """The IIT 3.0 mechanism MIP selection routes through
    resolve_ties.partitions and consults
    config.formalism.iit.mip_tie_resolution.

    Override to a registered-but-bogus strategy; the call must raise
    KeyError. The default preset value is ["PHI", "PARTITION_LEX"]
    which selects argmin raw phi (paper-canonical IIT 3.0 mechanism
    MIP).
    """
    from dataclasses import replace

    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.direction import Direction
    from pyphi.system import System

    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    bad = {**presets.iit3}
    bad["iit"] = replace(bad["iit"], mip_tie_resolution=["DEFINITELY_NOT_A_STRATEGY"])
    with config.override(**bad):
        sys = System.from_substrate(substrate, state, substrate.node_indices)
        with pytest.raises(KeyError):
            sys.find_mip(Direction.CAUSE, (1,), (2,))


def test_iit3_default_mip_tie_resolution_is_raw_phi():
    """presets.iit3's mip_tie_resolution is ["PHI", "PARTITION_LEX"]."""
    from pyphi.conf import presets

    assert presets.iit3["iit"].mip_tie_resolution == ["PHI", "PARTITION_LEX"]
