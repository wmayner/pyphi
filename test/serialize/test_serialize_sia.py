import json

import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
from pyphi.models.complex import ExcludedCandidate
from pyphi.models.explanation import NullResultReason
from pyphi.models.explanation import RunnerUp
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import NullCut
from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis
from pyphi.provenance import Provenance
from test.conftest import IIT_3_CONFIG

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_provenance_round_trips(fmt):
    obj = Provenance.capture()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_provenance_round_trips_estimator(fmt):
    obj = Provenance.capture(
        estimator={
            "regime": "perturbational",
            "model": "counts",
            "prior": 0.5,
            "n_transitions": 16,
            "n_states_observed": 8,
            "n_states_total": 8,
            "uncovered_state_count": 0,
        }
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert restored.estimator == obj.estimator


def test_provenance_decodes_payload_without_estimator():
    # Payloads written before the estimator field existed decode to None.
    doc = json.loads(serialize.dumps(Provenance.capture(), format="json"))
    doc["payload"].pop("estimator", None)
    restored = serialize.loads(json.dumps(doc).encode(), format="json")
    assert restored.estimator is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_excluded_candidate_round_trips(fmt):
    obj = ExcludedCandidate((0, 1), 0.5)
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_round_trips(fmt):
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is IIT3SystemIrreducibilityAnalysis


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_tie_peers(fmt):
    def make():
        return IIT3SystemIrreducibilityAnalysis(
            phi=0.5,
            partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
            node_indices=(0, 1),
            current_state=(1, 0),
        )

    a, b = make(), make()
    a.set_ties([a, b])
    b.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored.ties if t is not restored]
    assert len(peers) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_round_trips(fmt):
    obj = SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=NullCut((0, 1)),
        normalized_phi=0.25,
        current_state=(1, 0),
        node_indices=(0, 1),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is SystemIrreducibilityAnalysis
    # config degrades to a plain dict, matching the prior serializer.
    assert isinstance(restored.config, dict)


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_iit4_sia_round_trips(fmt):
    obj = NullSystemIrreducibilityAnalysis(node_indices=(0, 1))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is NullSystemIrreducibilityAnalysis


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_intrinsic_differentiation(fmt):
    obj = SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))
    obj.intrinsic_differentiation = {
        Direction.CAUSE: serialize.loads(
            serialize.dumps(obj.phi, format=fmt), format=fmt
        ),
        Direction.EFFECT: obj.normalized_phi,
    }
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert set(restored.intrinsic_differentiation) == {
        Direction.CAUSE,
        Direction.EFFECT,
    }


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_tie_peers(fmt):
    def make():
        return SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))

    a, b = make(), make()
    a.set_ties([a, b])
    b.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored.ties if t is not restored]
    assert len(peers) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_margins_round_trip(fmt):
    import pyphi
    from pyphi import examples

    with pyphi.config.override(progress_bars=False):
        sia = examples.basic_system().sia()
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert float(restored.partition_margin) == pytest.approx(float(sia.partition_margin))
    spec = restored.system_state.cause
    assert spec.runner_up_state == sia.system_state.cause.runner_up_state
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(
        float(sia.system_state.cause.runner_up_intrinsic_information)
    )
    assert restored.effectively_tied == sia.effectively_tied


def test_iit4_sia_loads_without_margin_fields():
    """Serialized results produced before margins existed decode with the
    margin fields at their defaults."""
    import json

    import pyphi
    from pyphi import examples
    from pyphi import serialize

    with pyphi.config.override(progress_bars=False):
        sia = examples.basic_system().sia()
    data = json.loads(serialize.dumps(sia, format="json"))

    def strip(obj):
        if isinstance(obj, dict):
            for key in (
                "partition_margin",
                "runner_up_state",
                "runner_up_intrinsic_information",
            ):
                obj.pop(key, None)
            for value in obj.values():
                strip(value)
        elif isinstance(obj, list):
            for item in obj:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.partition_margin is None
    assert restored.system_state.cause.runner_up_intrinsic_information is None
    assert restored.system_state.cause.state_margin is None
    assert not restored.effectively_tied
    # All pre-existing fields are untouched by the additions.
    assert float(restored.phi) == pytest.approx(float(sia.phi))
    assert restored.partition == sia.partition


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_runner_up_and_reasons(fmt):
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
        reasons=[NullResultReason.NO_SYSTEM],
        runner_up=RunnerUp(
            partition=DirectedBipartition(Direction.CAUSE, (1,), (0,)), phi=0.75
        ),
    )
    restored = round_trip(obj, fmt)
    assert restored.runner_up == obj.runner_up
    assert restored.reasons == [NullResultReason.NO_SYSTEM]


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_config_and_provenance(fmt):
    with IIT_3_CONFIG:
        obj = IIT3SystemIrreducibilityAnalysis(
            phi=0.5,
            partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
            node_indices=(0, 1),
            current_state=(1, 0),
        )
    restored = round_trip(obj, fmt)
    # config degrades to a plain dict, matching the IIT4 decoder.
    assert isinstance(restored.config, dict)
    assert restored.config["formalism"]["iit"]["version"] == "IIT_3_0"
    # provenance is the saved one, not freshly captured at load time.
    assert restored.provenance == obj.provenance


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_runner_up(fmt):
    obj = SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))
    obj.runner_up = RunnerUp(partition=NullCut((0, 1)), phi=0.75)
    restored = round_trip(obj, fmt)
    assert restored.runner_up == obj.runner_up


def test_iit3_sia_loads_without_new_fields():
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
    )
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("runner_up", "reasons", "config", "provenance"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.runner_up is None
    assert restored.reasons == []
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None
