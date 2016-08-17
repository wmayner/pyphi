#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py

import json
import tempfile

import numpy as np

from pyphi import compute, jsonify, models, network


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
    ]
    for o in objects:
        loaded = jsonify.loads(jsonify.dumps(o))
        assert loaded == o


def test_network_from_json(s):
    f = tempfile.NamedTemporaryFile(mode='wt')
    jsonify.dump(s.network, f)
    f.seek(0)
    loaded_network = network.from_json(f.name)
    assert loaded_network == s.network
    assert np.array_equal(loaded_network.node_labels, s.network.node_labels)


# TODO: these tests need to be fleshed out, they don't do much

def test_jsonify_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    jsonify.loads(jsonify.dumps(compute.big_mip(s)))


def test_jsonify_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = compute.complexes(s.network, s.state)
    jsonify.loads(jsonify.dumps(complexes))
