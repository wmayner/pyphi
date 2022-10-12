#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_new_big_phi.py

import json

import pytest

import pyphi
from pyphi.examples import EXAMPLES
from pyphi.jsonify import jsonify
from pyphi.new_big_phi import sia
from pyphi.compute.subsystem import ces

NETWORKS = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]

def expected_sia(example):
    SIA_PATH = f"test/data/sia/sia_{example}.json"
    
    with open(SIA_PATH) as f:
        expected = json.load(f)
    
    return expected

def expected_ces(example):
    CES_PATH = f"test/data/ces/ces_{example}.json"
    
    with open(CES_PATH) as f:
        expected = json.load(f)
    
    return expected

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "example_network", # TODO more parameters
    NETWORKS
)
def test_sia(example_network):
    example_func = EXAMPLES["subsystem"][example_network]
    actual = sia(example_func(), parallel=False)
    expected = expected_sia(example_network)
    
    actual = jsonify(actual)
    
    # node_labels.__id__ not expected to match
    del actual["node_labels"]["__id__"]
    del expected["node_labels"]["__id__"]
    
    assert actual == expected

# TODO failing via PyTest, but passing in notebook; nested equal dicts flagged not equal
@pytest.mark.parametrize(
    "example_network", # TODO more parameters
    NETWORKS
)
def test_compute_subsystem_ces(example_network, expected_ces):
    example_func = EXAMPLES["subsystem"][example_network]
    actual = ces(example_func())
    expected = expected_ces(example_network)
    
    actual = jsonify(actual)
    
    assert actual == expected

def test_phi_structure_match(example_network):
    assert False # TODO