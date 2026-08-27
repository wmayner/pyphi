import json

import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.models.explanation import NullResultReason
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_ria(direction=Direction.CAUSE, phi=0.3):
    return RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=direction,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_round_trips(fmt):
    obj = make_ria()
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert np.array_equal(restored.repertoire, obj.repertoire)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_partition_tie_peers(fmt):
    a = make_ria(phi=0.3)
    b = make_ria(phi=0.3)
    tied = (a, b)
    a._partition_ties = tied
    b._partition_ties = tied
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored._partition_ties if t is not restored]
    assert len(peers) == 1
    assert restored in peers[0]._partition_ties


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_state_tie_peers(fmt):
    a = make_ria(phi=0.3)
    b = make_ria(phi=0.3)
    tied = (a, b)
    a._state_ties = tied
    b._state_ties = tied
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored._state_ties if t is not restored]
    assert len(peers) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_mic_round_trips(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is MaximallyIrreducibleCause


@pytest.mark.parametrize("fmt", FORMATS)
def test_mie_round_trips(fmt):
    obj = MaximallyIrreducibleEffect(make_ria(Direction.EFFECT))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is MaximallyIrreducibleEffect


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_negative_signed_phi(fmt):
    obj = make_ria(phi=-0.25)
    assert float(obj.phi) == 0.0
    assert float(obj.signed_phi) == -0.25
    restored = round_trip(obj, fmt)
    assert float(restored.signed_phi) == -0.25
    assert float(restored.signed_normalized_phi) == float(obj.signed_normalized_phi)
    assert float(restored.phi) == 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_selectivity_and_reasons(fmt):
    obj = RepertoireIrreducibilityAnalysis(
        phi=0.3,
        direction=Direction.CAUSE,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
        selectivity=0.5,
        reasons=[NullResultReason.NO_PURVIEWS],
    )
    restored = round_trip(obj, fmt)
    assert restored.selectivity == 0.5
    assert restored.reasons == [NullResultReason.NO_PURVIEWS]


def test_ria_loads_without_new_fields():
    # Payloads written before these fields existed decode with the defaults.
    obj = make_ria()
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("signed_phi", "selectivity", "reasons"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert float(restored.signed_phi) == float(restored.phi)
    assert restored.selectivity is None
    assert restored.reasons is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_preserves_purview_ties(fmt):
    a = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    b = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    tied = (a, b)
    a._purview_ties = tied
    b._purview_ties = tied
    assert a.num_purview_ties == 1
    restored = round_trip(a, fmt)
    assert restored.num_purview_ties == 1
    peers = [t for t in restored._purview_ties if t is not restored]
    assert len(peers) == 1
    assert peers[0]._purview_ties is restored._purview_ties


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_nonmember_tie_tuple_round_trips(fmt):
    # set_purview_ties assigns the winner's tie tuple to state- and
    # partition-tie MICE that are not members of it; the round trip must
    # preserve the tuple instead of grafting the non-member in.
    winner = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    other = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    other._purview_ties = (winner,)  # `other` is not a member
    assert other.num_purview_ties == 0
    restored = round_trip(other, fmt)
    assert restored.num_purview_ties == 0
    assert len(restored._purview_ties) == 1
    assert all(t is not restored for t in restored._purview_ties)


def test_ces_purview_tie_counts_survive_round_trip():
    # End-to-end: in the basic system under the 2023 preset, the (2,) cause
    # carries the winner's tie tuple without being a member of it; its
    # num_purview_ties must not change from 0 to 1 across save/load.
    import pyphi
    from pyphi import examples
    from pyphi.conf import presets

    with pyphi.config.override(**presets.iit4_2023):
        ces = examples.basic_system().ces()
    restored = round_trip(ces, "msgpack")
    checked_nonmember = False
    for d, d2 in zip(ces.distinctions, restored.distinctions, strict=True):
        for side in ("cause", "effect"):
            m = getattr(d, side)
            m2 = getattr(d2, side)
            if m._purview_ties is None:
                assert m2._purview_ties is None
                continue
            assert len(m2._purview_ties) == len(m._purview_ties), (
                d.mechanism,
                side,
            )
            assert m2.num_purview_ties == m.num_purview_ties, (d.mechanism, side)
            if not any(t is m for t in m._purview_ties):
                checked_nonmember = True
    assert checked_nonmember, (
        "fixture no longer exercises a non-member tie tuple; "
        "pick a scenario where set_purview_ties assigns the winner's tuple"
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_not_computed_ties_round_trip(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    assert obj._purview_ties is None
    assert np.isnan(obj.num_purview_ties)
    restored = round_trip(obj, fmt)
    assert restored._purview_ties is None
    assert np.isnan(restored.num_purview_ties)


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_computed_no_ties_round_trip(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    obj._purview_ties = (obj,)
    restored = round_trip(obj, fmt)
    assert restored._purview_ties == (restored,)
    assert restored.num_purview_ties == 0


def test_mice_loads_without_tie_field_as_not_computed():
    # A payload without the field decodes as "ties not computed", not as
    # a claim of zero ties.
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    obj._purview_ties = (obj,)
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            o.pop("purview_tie_peers", None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored._purview_ties is None
    assert np.isnan(restored.num_purview_ties)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_normalized_phi_survives_cross_formalism_reload(fmt):
    """normalized_phi is stored, not recomputed from the ambient config.

    The RIA constructor derives normalized phi from the live
    ``distinction_phi_normalization`` option, so deserializing under a
    different formalism used to silently change the value (e.g. an IIT 3.0
    result reloaded under the 2026 default: 0.5 -> 0.1667).
    """
    from pyphi.conf import config
    from pyphi.conf import presets

    def make_discriminating_ria():
        # A partition cutting 2 connections, so NUM_CONNECTIONS_CUT (the
        # 2026 scheme) yields factor 1/2 while NONE (the IIT 3.0 scheme)
        # yields 1 -- the two schemes must disagree for this test to have
        # power against the recompute-from-ambient-config behavior.
        return RepertoireIrreducibilityAnalysis(
            phi=0.3,
            direction=Direction.CAUSE,
            mechanism=(0, 1),
            purview=(1, 2),
            partition=JointPartition(Part((0,), (1, 2)), Part((1,), ())),
            repertoire=np.ones((1, 2, 2)) / 4,
            partitioned_repertoire=np.ones((1, 2, 2)) / 4,
            mechanism_state=(1, 0),
            purview_state=(0, 1),
        )

    with config.override(**presets.iit3):
        obj = make_discriminating_ria()
        original = obj.normalized_phi
        blob = serialize.dumps(obj, format=fmt)
    with config.override(**presets.iit4_2026):
        # The schemes disagree on this fixture; without the stored value the
        # reload under the 2026 scheme would recompute a different number.
        assert make_discriminating_ria().normalized_phi != original
        restored = serialize.loads(blob, format=fmt)
    assert restored.normalized_phi == original
    assert restored.signed_normalized_phi == obj.signed_normalized_phi
