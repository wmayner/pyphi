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

example_subsystems = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]

def expected_sia(example):
    SIA_PATH = f"test/data/sia_{example}.json"
    
    with open(SIA_PATH) as f:
        expected = json.load(f)
    
    return expected

def expected_ces(example):
    CES_PATH = f"test/data/sia_{example}.json"
    
    with open(CES_PATH) as f:
        expected = json.load(f)
    
    return expected

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "example_subsystem" # TODO more parameters
    [example_subsystems]
)
def test_sia(example_subsystem):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual = sia(example_func(), parallel=False)
    expected = expected_sia(example_subsystem)
    
    actual = jsonify(actual)
    
    # node_labels.__id__ not expected to match
    del actual["node_labels"]["__id__"]
    del expected[example_subsystem]["node_labels"]["__id__"]
    
    assert actual == expected

# TODO failing via PyTest, but passing in notebook; nested equal dicts flagged not equal
@pytest.mark.parametrize(
    "example_subsystem", # TODO more parameters
    example_subsystems
)
def test_compute_subsystem_ces(example_subsystem, expected_ces):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual = ces(example_func())
    expected = expected_ces(example_subsystem)
    
    actual = jsonify(actual)
    
    assert actual == expected

def test_phi_structure_match(example_subsystem):
    assert False # TODO