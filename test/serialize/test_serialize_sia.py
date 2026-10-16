import json
from pathlib import Path

import pytest

from pyphi import serialize
from pyphi.conf.snapshot import ConfigSnapshot
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
    assert isinstance(restored.config, ConfigSnapshot)


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
    assert float(spec.runner_up_intrinsic_specification) == pytest.approx(
        float(sia.system_state.cause.runner_up_intrinsic_specification)
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
                "runner_up_intrinsic_specification",
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
    assert restored.system_state.cause.runner_up_intrinsic_specification is None
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
    assert isinstance(restored.config, ConfigSnapshot)
    assert restored.config.formalism.iit.version == "IIT_3_0"
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


RC3_FIXTURE = (
    Path(__file__).parents[1] / "data" / "serialize" / "rc3_fig1a_ab_analysis.json"
)


def test_rc3_saved_state_specification_still_loads(tmp_path):
    """An analysis saved by 2.0.0rc3, whose state specifications carry
    ``intrinsic_information`` / ``runner_up_intrinsic_information`` and whose
    distinctions carry ``purview_state``, loads under the current names and
    saves under them."""
    import pyphi
    from pyphi import examples

    old = RC3_FIXTURE.read_text()
    assert '"runner_up_intrinsic_information"' in old
    assert '"purview_state"' in old

    loaded = serialize.load(RC3_FIXTURE)
    with pyphi.config.override(**pyphi.iit4_2026):
        fresh = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1)
        )

    assert loaded.phi == pytest.approx(fresh.phi)
    assert loaded.big_phi == pytest.approx(fresh.big_phi)
    for direction in Direction.both():
        spec = loaded.sia.system_state[direction]
        expected = fresh.sia.system_state[direction]
        assert spec.intrinsic_specification == pytest.approx(
            expected.intrinsic_specification
        )
        assert spec.runner_up_intrinsic_specification == pytest.approx(
            expected.runner_up_intrinsic_specification
        )
    assert loaded.sia.system_state.cause.intrinsic_specification == pytest.approx(
        0.843228, abs=1e-6
    )
    for old_d, new_d in zip(
        loaded.ces.distinctions, fresh.ces.distinctions, strict=True
    ):
        for direction in Direction.both():
            old_mice, new_mice = old_d.mice(direction), new_d.mice(direction)
            assert old_mice.current_state == new_mice.current_state
            assert old_mice.specified_state.state == new_mice.specified_state.state
            assert old_mice.specified_state.intrinsic_specification == pytest.approx(
                new_mice.specified_state.intrinsic_specification
            )
        assert old_d.cause_purview_label == new_d.cause_purview_label
        assert old_d.effect_purview_label == new_d.effect_purview_label

    resaved = tmp_path / "resaved.json"
    serialize.save(loaded, resaved)
    text = resaved.read_text()
    assert '"intrinsic_information"' not in text
    assert '"runner_up_intrinsic_information"' not in text
    assert '"purview_state"' not in text
    assert '"intrinsic_specification"' in text


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_keeps_the_system_state_tie_family(fmt):
    """A tied SIA's system-state tie family holds an equal copy of the
    selected state; a round trip must not add it a second time."""
    import pyphi
    from pyphi import examples

    with pyphi.config.override(**pyphi.iit4_2023):
        sia = pyphi.analyze(
            examples.xor_substrate(), (0, 0, 0), compute="sia", progress=False
        )
    families = {
        d: [tuple(t.state) for t in sia.system_state[d].ties] for d in Direction.both()
    }
    assert len(families[Direction.CAUSE]) > 1
    restored = round_trip(sia, fmt)
    for direction in Direction.both():
        assert [tuple(t.state) for t in restored.system_state[direction].ties] == (
            families[direction]
        )
