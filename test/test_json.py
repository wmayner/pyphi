#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pyphi


def test_json_make_encodable_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    bm = pyphi.compute.big_mip(s)
    encodable = pyphi.json.make_encodable(bm)
    # Try encoding and decoding.
    json.loads(json.dumps(encodable))


def test_json_encode_big_mip(s, flushcache, restore_fs_cache):
    flushcache()
    # Try decoding.
    pyphi.json.dumps(pyphi.compute.big_mip(s))


def test_json_make_encodable_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    complexes = pyphi.compute.complexes(s.network, s.state)
    encodable = pyphi.json.make_encodable(complexes)
    # Try encoding and decoding.
    json.loads(json.dumps(encodable))


def test_json_encode_complexes(s, flushcache, restore_fs_cache):
    flushcache()
    # Try decoding.
    pyphi.json.dumps(pyphi.compute.complexes(s.network, s.state))
