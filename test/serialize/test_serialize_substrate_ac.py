import json

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi import serialize
from pyphi.actual import Transition
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.models.complex import ExcludedCandidate
from pyphi.models.explanation import NullResultReason
from pyphi.substrate import Substrate

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_transition():
    sub = examples.actual_causation_substrate()
    return Transition(
        sub,
        before_state=(1, 1),
        after_state=(1, 1),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_round_trips(fmt):
    obj = examples.basic_substrate()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_system_round_trips(fmt):
    obj = examples.basic_system()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_round_trips(fmt):
    obj = make_transition()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_account_round_trips(fmt):
    obj = actual.account(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is type(obj)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_round_trips(fmt):
    obj = actual.sia(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_complex_round_trips(fmt):
    obj = Complex(
        sia=NullSystemIrreducibilityAnalysis(node_indices=(0, 1)),
        substrate=examples.basic_substrate(),
        is_maximal=True,
        excluded=(ExcludedCandidate((0,), 0.1),),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is Complex


@pytest.mark.parametrize("fmt", FORMATS)
def test_kary_substrate_round_trips(fmt):
    obj = examples.gomez_p53_mdm2_substrate()  # ternary p53 + binary Mdm2 units
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert list(restored.node_labels) == list(obj.node_labels)
    assert restored.factored_tpm.state_space == obj.factored_tpm.state_space


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_preserves_noise_background(fmt):
    # OR gate driven by a noised background unit: the EFFECT ratio is
    # nonzero only when noise_background survives the round-trip.
    substrate = Substrate(np.array([[0, 0], [1, 1], [1, 1], [1, 1]]))
    obj = Transition(substrate, (1, 1), (1, 1), (0,), (1,), noise_background=True)
    restored = round_trip(obj, fmt)
    assert restored.noise_background is True
    assert restored == obj
    original_ratio = obj._ratio(Direction.EFFECT, (0,), (1,))
    assert restored._ratio(Direction.EFFECT, (0,), (1,)) == original_ratio
    assert original_ratio != 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_ac_sia_round_trips(fmt):
    sia = actual._null_ac_sia(make_transition(), Direction.CAUSE)
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert len(restored.account) == 0
    assert len(restored.partitioned_account) == 0


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_ria_preserves_node_labels(fmt):
    s = actual.sia(make_transition())
    link = next(iter(s.account))
    assert link.ria.node_labels is not None
    restored = round_trip(s, fmt)
    rlink = next(iter(restored.account))
    assert rlink.ria.node_labels == link.ria.node_labels


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_preserves_reasons_ties_config_provenance(fmt):
    t = make_transition()
    a = actual._null_ac_sia(t, Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM])
    b = actual._null_ac_sia(t, Direction.CAUSE, alpha=0.5)
    a.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored.reasons == [NullResultReason.NO_SYSTEM]
    peers = [p for p in restored.ties if p is not restored]
    assert len(peers) == 1
    assert isinstance(restored.config, dict)
    assert restored.provenance == a.provenance


def test_ac_sia_loads_without_new_fields():
    sia = actual._null_ac_sia(
        make_transition(), Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM]
    )
    data = json.loads(serialize.dumps(sia, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("reasons", "config", "provenance", "tie_peers"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.reasons == []
    assert restored.ties == (restored,)
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None
