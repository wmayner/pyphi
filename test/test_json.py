#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py

import numpy as np

import pyphi


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
    assert answer == pyphi.jsonify.loads(pyphi.jsonify.dumps(x))


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
    assert answer == pyphi.jsonify.loads(pyphi.jsonify.dumps(x))


# TODO: these tests need to be fleshed out, they don't do much


def test_jsonify_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    pyphi.jsonify.loads(pyphi.jsonify.dumps(pyphi.compute.big_mip(s)))


def test_jsonify_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = pyphi.compute.complexes(s.network, s.state)
    pyphi.jsonify.loads(pyphi.jsonify.dumps(complexes))
