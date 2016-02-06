#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py

import tempfile

import numpy as np

from pyphi import compute, jsonify, network


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
    assert answer == jsonify.loads(jsonify.dumps(x))


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
    assert answer == jsonify.loads(jsonify.dumps(x))


def test_jsonify_network(s):
    loaded = jsonify.loads(jsonify.dumps(s.network))
    assert np.array_equal(loaded['tpm'], s.network.tpm)
    assert np.array_equal(loaded['cm'], s.network.connectivity_matrix)
    assert loaded['size'] == s.network.size


def test_network_from_json(s):
    f = tempfile.NamedTemporaryFile(mode='wt')
    jsonify.dump(s.network, f)
    f.seek(0)
    assert network.from_json(f.name) == s.network


# TODO: these tests need to be fleshed out, they don't do much

def test_jsonify_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    jsonify.loads(jsonify.dumps(compute.big_mip(s)))


def test_jsonify_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = compute.complexes(s.network, s.state)
    jsonify.loads(jsonify.dumps(complexes))
