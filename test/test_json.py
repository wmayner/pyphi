#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_json.py


import pyphi


def test_json_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    pyphi.jsonify.loads(pyphi.jsonify.dumps(pyphi.compute.big_mip(s)))


def test_json_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = pyphi.compute.complexes(s.network, s.state)
    pyphi.jsonify.loads(pyphi.jsonify.dumps(complexes))
