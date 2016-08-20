#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py

import json
import tempfile

import pytest
import numpy as np

from pyphi import compute, exceptions, jsonify, models, network


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


def test_json_deserialization(s):
    objects = [
        s.network,  # Network
        s,  # Subsystem
        models.Bipartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        s.concept((1, 2)),
        s.concept((1,)),
        compute.constellation(s),
        compute.big_mip(s)
    ]
    for o in objects:
        loaded = jsonify.loads(jsonify.dumps(o))
        assert loaded == o


def test_deserialization_memoizes_duplicate_objects(s):
    big_mip = compute.big_mip(s)

    def check_subsystems(big_mip):
        assert (big_mip.subsystem is
            big_mip.unpartitioned_constellation[0].subsystem)

    check_subsystems(big_mip)
    loaded = jsonify.loads(jsonify.dumps(big_mip))
    check_subsystems(loaded)


def test_network_from_json(s):
    f = tempfile.NamedTemporaryFile(mode='wt')
    jsonify.dump(s.network, f)
    f.seek(0)
    loaded_network = network.from_json(f.name)
    assert loaded_network == s.network
    assert np.array_equal(loaded_network.node_labels, s.network.node_labels)


def test_version_check_during_deserialization(s):
    string = jsonify.dumps(s)

    # Change the version
    _obj = json.loads(string)
    _obj[jsonify.VERSION_KEY] = '0.1.bogus'
    string = json.dumps(_obj)

    with pytest.raises(exceptions.JSONVersionError):
        jsonify.loads(string)
