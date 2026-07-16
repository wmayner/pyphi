"""Document-level node-labels frame: one label table per document, inherited
on decode, with per-object overrides for heterogeneous documents."""

import msgspec
import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.labels import NodeLabels
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.serialize import convert

FORMATS = ["json", "msgpack"]

LABELS = NodeLabels(("A", "B"), (0, 1))
OTHER = NodeLabels(("X", "Y"), (0, 1))


def make_labeled_ria(node_labels: NodeLabels | None = LABELS, phi=0.3):
    return RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=Direction.CAUSE,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
        node_labels=node_labels,
    )


def test_document_claims_one_frame():
    data = serialize.dumps(make_labeled_ria())
    doc = msgspec.json.decode(data)
    assert doc["node_labels"] is not None
    assert doc["payload"]["node_labels"] is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_labels_round_trip_via_frame(fmt):
    obj = make_labeled_ria()
    restored = serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)
    assert restored == obj
    assert tuple(restored.node_labels) == tuple(LABELS)


def test_unlabeled_object_round_trips_without_frame():
    obj = make_labeled_ria(node_labels=None)
    data = serialize.dumps(obj)
    doc = msgspec.json.decode(data)
    assert doc["node_labels"] is None
    restored = serialize.loads(data)
    assert restored.node_labels is None


def test_old_format_per_object_labels_still_load():
    # Documents written before the frame carried labels on every struct and
    # no envelope frame; encoding outside a document context reproduces that
    # layout exactly.
    payload = convert.to_schema(make_labeled_ria())
    assert payload.node_labels is not None  # inline, old-style
    doc = serialize._Document(format_version=1, payload=payload)
    restored = serialize.loads(msgspec.json.encode(doc))
    assert tuple(restored.node_labels) == tuple(LABELS)


def test_heterogeneous_labels_survive_as_overrides():
    a = make_labeled_ria(node_labels=LABELS)
    b = make_labeled_ria(node_labels=OTHER)
    tied = (a, b)
    a._partition_ties = tied
    b._partition_ties = tied
    restored = serialize.loads(serialize.dumps(a))
    assert tuple(restored.node_labels) == tuple(LABELS)
    peer = next(t for t in restored._partition_ties if t is not restored)
    assert tuple(peer.node_labels) == tuple(OTHER)


def test_loads_node_labels_override_wins():
    data = serialize.dumps(make_labeled_ria())
    restored = serialize.loads(data, node_labels=OTHER)
    assert tuple(restored.node_labels) == ("X", "Y")
