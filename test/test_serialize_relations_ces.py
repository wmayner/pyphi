import numpy as np
import pytest

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
