import json

import numpy as np
import pytest

import pyphi
from pyphi import serialize
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullCauseEffectStructure
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.distinction import Distinction
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.relations import ConcreteRelations
from pyphi.relations import NullRelations
from pyphi.relations import Relation

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_ria(direction, node):
    return RepertoireIrreducibilityAnalysis(
        phi=0.3,
        direction=direction,
        mechanism=(node,),
        purview=(1,),
        partition=JointPartition(Part((node,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
    )


def make_distinction(node=0):
    return Distinction(
        mechanism=(node,),
        cause=MaximallyIrreducibleCause(make_ria(Direction.CAUSE, node)),
        effect=MaximallyIrreducibleEffect(make_ria(Direction.EFFECT, node)),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_relation_round_trips(fmt):
    rel = Relation([make_distinction(0), make_distinction(1)])
    restored = round_trip(rel, fmt)
    assert restored == rel


@pytest.mark.parametrize("fmt", FORMATS)
def test_concrete_relations_round_trips(fmt):
    rel = Relation([make_distinction(0), make_distinction(1)])
    obj = ConcreteRelations([rel])
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is ConcreteRelations


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_relations_round_trips(fmt):
    obj = NullRelations()
    restored = round_trip(obj, fmt)
    assert type(restored) is NullRelations
    assert len(list(restored)) == 0


def make_ces():
    d0, d1 = make_distinction(0), make_distinction(1)
    distinctions = ResolvedDistinctions([d0, d1])
    # Build the relation from the same distinction objects so the encoder's
    # identity-based index lookup is exercised.
    rel = Relation([d0, d1])
    relations = ConcreteRelations([rel])
    sia = NullSystemIrreducibilityAnalysis(node_indices=(0, 1))
    return CauseEffectStructure(sia=sia, distinctions=distinctions, relations=relations)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ces_round_trips(fmt):
    obj = make_ces()
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is CauseEffectStructure


def test_ces_relations_are_stored_as_index_refs():
    # The normalized form references distinctions by index rather than
    # embedding them inside each relation.
    obj = make_ces()
    data = serialize.dumps(obj, format="json")
    assert b"distinction_indices" in data
    assert b"relation_ref" in data


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_ces_round_trips(fmt):
    obj = NullCauseEffectStructure()
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is NullCauseEffectStructure


@pytest.mark.parametrize("fmt", FORMATS)
def test_ces_preserves_config_and_provenance(fmt):
    with pyphi.config.override(precision=7):
        obj = make_ces()
    restored = round_trip(obj, fmt)
    assert isinstance(restored.config, dict)
    assert restored.config["numerics"]["precision"] == 7
    assert restored.provenance == obj.provenance


def test_ces_loads_without_config_and_provenance():
    # Strip only the CES-level keys: payloads written before these fields
    # existed lack them at the CES level but carry them on the embedded SIA.
    obj = make_ces()
    data = json.loads(serialize.dumps(obj, format="json"))
    data["payload"].pop("config", None)
    data["payload"].pop("provenance", None)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None


def test_ces_analytical_relations_share_decoded_distinctions():
    """The decoded CES's analytical relations wrap the structure's own
    distinctions object, preserving wrapper type and identity sharing."""
    import dataclasses

    import pyphi
    from pyphi import examples
    from pyphi import serialize
    from pyphi.conf import presets
    from pyphi.relations import AnalyticalRelations

    with pyphi.config.override(**presets.iit4_2023):
        ces = examples.basic_system().ces()
    analytical = dataclasses.replace(
        ces, relations=AnalyticalRelations(ces.distinctions)
    )
    loaded = serialize.loads(serialize.dumps(analytical))
    assert isinstance(loaded.relations, AnalyticalRelations)
    assert loaded.relations.distinctions is loaded.distinctions


@pytest.mark.parametrize("formalism", ["IIT_4_0_2026", "IIT_4_0_2023", "IIT_3_0"])
def test_analysis_roundtrips(formalism):
    """`analyze`'s own return type round-trips, under every formalism.

    IIT 4.0 wraps the distinctions in a Φ-structure while IIT 3.0's cause-effect
    structure is the bare distinction sequence, so the `ces` field is a union.
    """
    pyphi.config.progress_bars = False
    result = pyphi.analyze(
        pyphi.examples.basic_substrate(), (1, 0, 0), formalism=formalism
    )
    restored = serialize.loads(serialize.dumps(result))
    assert type(restored) is pyphi.Analysis
    assert restored.phi == result.phi
    assert restored.ces == result.ces
    assert restored.system == result.system
    if formalism != "IIT_3_0":
        assert restored.big_phi == result.big_phi


def test_analysis_saves_and_loads(tmp_path):
    pyphi.config.progress_bars = False
    result = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 0, 0))
    path = tmp_path / "analysis.pyphi.gz"
    result.save(path)
    assert pyphi.Analysis.load(path).phi == result.phi
