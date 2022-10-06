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

@pytest.fixture
def expected_sia():
    cases = {}
    
    for example in example_subsystems:
        with open(f"test/data/sia/sia_{example}.json") as f:
            cases[example] = json.load(f)
    
    return cases

@pytest.fixture
def expected_ces():
    cases = {}
    
    for example in example_subsystems:
        with open(f"test/data/ces/ces_{example}.json") as f:
            cases[example] = json.load(f)
    
    return cases

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "example_subsystem", # TODO more parameters
    example_subsystems
)
def test_sia(example_subsystem, expected_sia):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual_sia = sia(example_func(), parallel=False)
    
    actual_sia = jsonify(actual_sia)
    
    # node_labels.__id__ not expected to match
    del actual_sia["node_labels"]["__id__"]
    del expected_sia[example_subsystem]["node_labels"]["__id__"]
    
    assert actual_sia == expected_sia[example_subsystem]

# TODO failing via PyTest, but passing in notebook; nested equal dicts flagged not equal
@pytest.mark.parametrize(
    "example_subsystem", # TODO more parameters
    example_subsystems
)
def test_compute_subsystem_ces(example_subsystem, expected_ces):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual_ces = ces(example_func())
    
    actual_ces = jsonify(actual_ces)
    
    for key, value in actual_ces.items():
        actual_ces[key] = jsonify(value)
    
    assert actual_ces == expected_ces[example_subsystem]

def test_phi_structure_match(example_subsystem):
    assert False # TODO