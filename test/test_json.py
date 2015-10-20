#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py


import pyphi


def test_jsonify_native(s):
    x = {
        'list': [1, 2.0, 3],
        'tuple': (1, 2, 3),
        'bool': [True, False],
        'null': None
    }
    pyphi.jsonify.loads(pyphi.jsonify.dumps(x))


def test_jsonify_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    pyphi.jsonify.loads(pyphi.jsonify.dumps(pyphi.compute.big_mip(s)))


def test_jsonify_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = pyphi.compute.complexes(s.network, s.state)
    pyphi.jsonify.loads(pyphi.jsonify.dumps(complexes))
