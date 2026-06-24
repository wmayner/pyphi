"""Round-trip tests for tie metadata through ``to_json`` / ``from_json``.

Covers ``StateSpecification._ties``, ``RepertoireIrreducibilityAnalysis``
``_partition_ties`` / ``_state_ties``, and
``SystemIrreducibilityAnalysis._ties``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyphi import jsonify
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.warnings import PyPhiWarning


def _make_state_spec(
    state: tuple[int, ...] = (0, 1), ii: float = 0.5
) -> StateSpecification:
    return StateSpecification(
        direction=Direction.EFFECT,
        purview=(0, 1),
        state=state,
        intrinsic_information=PyPhiFloat(ii),
        repertoire=np.array([0.25, 0.25, 0.25, 0.25]),
        unconstrained_repertoire=np.array([0.25, 0.25, 0.25, 0.25]),
    )


def _make_ria(
    phi: float = 0.5,
    mechanism: tuple[int, ...] = (0,),
    purview: tuple[int, ...] = (1,),
    partition: JointPartition | None = None,
) -> RepertoireIrreducibilityAnalysis:
    if partition is None:
        partition = JointPartition(
            Part(mechanism=(0,), purview=(1,)),
            Part(mechanism=(), purview=()),
        )
    return RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=Direction.EFFECT,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        repertoire=np.array([0.5, 0.5]),
        partitioned_repertoire=np.array([0.5, 0.5]),
    )


def _make_sia(phi: float = 1.0) -> SystemIrreducibilityAnalysis:
    return SystemIrreducibilityAnalysis(
        phi=phi,
        partition=NullCut(indices=(0, 1)),
        normalized_phi=phi,
        current_state=(0, 1),
        node_indices=(0, 1),
    )


def _round_trip(obj):
    return jsonify.loads(jsonify.dumps(obj))


class TestStateSpecificationTieRoundTrip:
    def test_no_ties_round_trips_with_empty_ties(self):
        spec = _make_state_spec()
        assert spec.ties == ()
        restored = _round_trip(spec)
        assert restored == spec
        assert restored.ties == ()

    def test_ties_preserve_full_tied_set(self):
        spec_a = _make_state_spec(state=(0, 0), ii=0.5)
        spec_b = _make_state_spec(state=(0, 1), ii=0.5)
        spec_c = _make_state_spec(state=(1, 0), ii=0.5)
        tied = (spec_a, spec_b, spec_c)
        for s in tied:
            s.set_ties(tied)
        restored = _round_trip(spec_a)
        assert restored == spec_a
        assert len(restored.ties) == len(tied)
        restored_states = {t.state for t in restored.ties}
        assert restored_states == {(0, 0), (0, 1), (1, 0)}

    def test_round_trip_does_not_emit_pyphi_warning(self):
        spec_a = _make_state_spec(state=(0, 0))
        spec_b = _make_state_spec(state=(0, 1))
        spec_a.set_ties((spec_a, spec_b))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyPhiWarning)
            _round_trip(spec_a)


class TestRiaTieRoundTrip:
    def test_no_ties_round_trips_with_self_singleton(self):
        ria = _make_ria()
        assert ria.partition_ties == (ria,)
        assert ria.state_ties == (ria,)
        restored = _round_trip(ria)
        assert restored == ria
        assert restored.partition_ties == (restored,)
        assert restored.state_ties == (restored,)

    def test_partition_ties_round_trip_preserves_peers(self):
        ria_a = _make_ria(mechanism=(0,))
        partition_b = JointPartition(
            Part(mechanism=(0,), purview=()),
            Part(mechanism=(), purview=(1,)),
        )
        ria_b = _make_ria(mechanism=(0,), partition=partition_b)
        ria_a.set_partition_ties((ria_a, ria_b))
        restored = _round_trip(ria_a)
        assert len(restored.partition_ties) == 2
        assert restored in restored.partition_ties
        peer = next(t for t in restored.partition_ties if t is not restored)
        assert peer.partition == ria_b.partition

    def test_state_ties_round_trip_preserves_peers(self):
        ria_a = _make_ria(mechanism=(0,))
        ria_b = _make_ria(mechanism=(1,))
        ria_a.set_state_ties((ria_a, ria_b))
        restored = _round_trip(ria_a)
        assert len(restored.state_ties) == 2
        peer = next(t for t in restored.state_ties if t is not restored)
        assert peer.mechanism == (1,)

    def test_partition_and_state_ties_round_trip_independently(self):
        ria_a = _make_ria(mechanism=(0,), purview=(1,))
        part_b = JointPartition(
            Part(mechanism=(0,), purview=()),
            Part(mechanism=(), purview=(1,)),
        )
        ria_partition_peer = _make_ria(mechanism=(0,), purview=(1,), partition=part_b)
        ria_state_peer = _make_ria(mechanism=(1,), purview=(0,))
        ria_a.set_partition_ties((ria_a, ria_partition_peer))
        ria_a.set_state_ties((ria_a, ria_state_peer))
        restored = _round_trip(ria_a)
        assert len(restored.partition_ties) == 2
        assert len(restored.state_ties) == 2
        partition_peer = next(t for t in restored.partition_ties if t is not restored)
        state_peer = next(t for t in restored.state_ties if t is not restored)
        assert partition_peer.partition == part_b
        assert state_peer.mechanism == (1,)

    def test_round_trip_does_not_emit_pyphi_warning(self):
        ria = _make_ria()
        peer = _make_ria(mechanism=(1,))
        ria.set_partition_ties((ria, peer))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyPhiWarning)
            _round_trip(ria)


class TestSiaTieRoundTrip:
    def test_no_ties_round_trips_with_self_singleton(self):
        sia = _make_sia()
        assert sia.ties == [sia]
        restored = _round_trip(sia)
        assert restored == sia
        assert restored.ties == [restored]

    def test_ties_preserve_full_tied_set(self):
        sia_a = _make_sia(phi=1.0)
        sia_b_partition = DirectedBipartition(Direction.EFFECT, (0,), (1,))
        sia_b = SystemIrreducibilityAnalysis(
            phi=1.0,
            partition=sia_b_partition,
            normalized_phi=1.0,
            current_state=(0, 1),
            node_indices=(0, 1),
        )
        tied = (sia_a, sia_b)
        sia_a.set_ties(tied)
        sia_b.set_ties(tied)
        restored = _round_trip(sia_a)
        assert len(restored.ties) == 2
        # Original is one of the ties (regardless of order/identity).
        partitions = {tuple(t.partition.lex_key()) for t in restored.ties}
        assert tuple(NullCut(indices=(0, 1)).lex_key()) in partitions
        assert tuple(sia_b_partition.lex_key()) in partitions

    def test_round_trip_does_not_emit_pyphi_warning(self):
        sia_a = _make_sia(phi=1.0)
        sia_b = SystemIrreducibilityAnalysis(
            phi=1.0,
            partition=DirectedBipartition(Direction.EFFECT, (0,), (1,)),
            normalized_phi=1.0,
            current_state=(0, 1),
            node_indices=(0, 1),
        )
        sia_a.set_ties((sia_a, sia_b))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyPhiWarning)
            _round_trip(sia_a)


@pytest.mark.parametrize(
    "constructor",
    [_make_state_spec, _make_ria, _make_sia],
)
def test_dumps_does_not_emit_tie_serialization_warning(constructor):
    """Degenerate self-ties (the no-ties default) do not trigger a warning."""
    obj = constructor()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyPhiWarning)
        jsonify.dumps(obj)
