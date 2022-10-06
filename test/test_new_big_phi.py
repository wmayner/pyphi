#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_new_big_phi.py

import json
from copy import copy

import pytest

import pyphi
from pyphi.examples import EXAMPLES
from pyphi.jsonify import jsonify
from pyphi.new_big_phi import sia
from pyphi.compute.subsystem import ces

example_sia_subsystems = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]
example_ces_subsystems = ["basic", "fig4", "fig5a", "fig5b", "grid3", "rule110", "xor"]

@pytest.fixture
def expected_sia():
    cases = {}
    
    for example in example_sia_subsystems:
        with open(f"test/data/sia/sia_{example}.json") as f:
            cases[example] = json.load(f)
    
    return cases

@pytest.fixture
def expected_ces():
    cases = {}
    
    for example in example_ces_subsystems:
        with open(f"test/data/ces/ces_{example}.json") as f:
            cases[example] = json.load(f)
    
    return cases

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "example_subsystem", # TODO more parameters
    example_sia_subsystems
)
def test_sia(example_subsystem, expected_sia):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual_sia = sia(example_func(), parallel=False)
    
    # convert SIA object to JSON format
    actual_sia = jsonify(actual_sia)
    
    del actual_sia["node_labels"]["__id__"]
    del expected_sia[example_subsystem]["node_labels"]["__id__"]
    
    assert actual_sia == expected_sia[example_subsystem]

@pytest.mark.parametrize(
    "example_subsystem", # TODO more parameters
    example_ces_subsystems
)
def test_compute_subsystem_ces(example_subsystem, expected_ces):
    example_func = EXAMPLES["subsystem"][example_subsystem]
    actual_ces = ces(example_func())
    
    # convert CES object to JSON format
    actual_ces = copy(actual_ces.__dict__)
    jsonify(actual_ces)
    
    for key, value in actual_ces.items():
        actual_ces[key] = jsonify(value)
    
    assert actual_ces.keys() == expected_ces[example_subsystem].keys()
        
    for key, value in expected_ces[example_subsystem].items():
        # ignore node_labels.__id__
        if key == "node_labels":
            for attr in value:
                if attr != "__id__":
                    assert actual_ces[key][attr] == value[attr]
        else:
            assert actual_ces[key] == expected_ces[example_subsystem][key]

def test_relations():
    assert False == True # TODO