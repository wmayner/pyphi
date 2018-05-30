#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py

import json
import tempfile

import numpy as np
import pytest

from pyphi import (Direction, actual, compute, config, exceptions, jsonify,
                   labels, models, network)
from test_actual import transition


def test_jsonify_native():
    x = {
        'list': [1, 2.0, 3],
        'tuple': (1, 2, 3),
        'bool': [True, False],
        'null': None
    }
    answer = {
        'list': [1, 2.0, 3],
        'tuple': [1, 2, 3],
        'bool': [True, False],
        'null': None
    }
    assert answer == json.loads(jsonify.dumps(x))


def test_jsonify_numpy():
    x = {
        'ndarray': np.array([1, 2]),
        'np.int32': np.int32(1),
        'np.int64': np.int64(2),
        'np.float64': np.float64(3),
    }
    answer = {
        'ndarray': [1, 2],
        'np.int32': 1,
        'np.int64': 2,
        'np.float64': 3.0,
    }
    assert answer == json.loads(jsonify.dumps(x))


def test_json_deserialization(s, transition):
    objects = [
        Direction.CAUSE,
        s.network,  # Network
        s,  # Subsystem
        models.Bipartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        models.KPartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        models.Tripartition(models.Part((0,), ()), models.Part((1,), (2, 3)),
                            models.Part((3,), (4,))),
        models.Cut((0,), (2,)),
        models.NullCut((0, 1)),
        models.KCut(Direction.CAUSE,
                    models.KPartition(models.Part((0,), ()),
                                      models.Part((1,), (2, 3)))),
        s.concept((1, 2)),
        s.concept((1,)),
        compute.ces(s),
        compute.sia(s),
        transition,
        transition.find_actual_cause((0,), (0,)),
        actual.account(transition),
        actual.sia(transition),
        labels.NodeLabels('AB', (0, 1))

    ]
    for o in objects:
        loaded = jsonify.loads(jsonify.dumps(o))
        assert loaded == o


def test_json_deserialization_non_pyphi_clasess():
    class OtherObject:
        def __init__(self, x):
            self.x = x

    loaded = jsonify.loads(jsonify.dumps(OtherObject(1)))
    assert loaded == {'x': 1}


def test_deserialization_memoizes_duplicate_objects(s):
    with config.override(PARALLEL_CUT_EVALUATION=True):
        sia = compute.sia(s)

    loaded = jsonify.loads(jsonify.dumps(sia))

    l1 = loaded.subsystem
    l2 = loaded.ces.subsystem
    assert l1 == l2
    assert hash(l1) == hash(l2)
    assert l1 is l2


@pytest.fixture
def network_file(standard):
    f = tempfile.NamedTemporaryFile(mode='w+')
    jsonify.dump(standard, f)
    f.seek(0)
    return f


def test_load(network_file, standard):
    assert jsonify.load(network_file) == standard


def test_network_from_json(network_file, standard):
    loaded_network = network.from_json(network_file.name)
    assert loaded_network == standard
    assert np.array_equal(loaded_network.node_labels, standard.node_labels)


def test_version_check_during_deserialization(s):
    string = jsonify.dumps(s)

    # Change the version
    _obj = json.loads(string)
    _obj[jsonify.VERSION_KEY] = '0.1.bogus'
    string = json.dumps(_obj)

    with pytest.raises(exceptions.JSONVersionError):
        jsonify.loads(string)
