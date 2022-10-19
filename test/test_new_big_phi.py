#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_new_big_phi.py

import json
from pyrsistent import v

import pytest

import pyphi
from pyphi.examples import EXAMPLES
from pyphi.jsonify import jsonify
from pyphi.new_big_phi import sia
from pyphi.compute.subsystem import ces
from pyphi.relations import relations

NETWORKS = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]

def expected_sia(example):
    SIA_PATH = f"test/data/new_big_phi/sia/sia_{example}.json"
    
    with open(SIA_PATH) as f:
        expected = json.load(f)
    
    return expected

def expected_ces(example):
    CES_PATH = f"test/data/new_big_phi/ces/ces_{example}.json"
    
    with open(CES_PATH) as f:
        expected = json.load(f)
    
    return expected

def expected_relations(example):
    RELATIONS_PATH = f"test/data/new_big_phi/relations/relations_{example}.json"
    
    with open(RELATIONS_PATH) as f:
        expected = json.load(f)
    
    return expected

def remove_ids(dct: dict):
    has_id = False
    
    for key, value in dct.items():
        if isinstance(value, dict):
            remove_ids(value)
            
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_ids(item)
            
        if key == "__id__":
            has_id = True
            
    if has_id:
        del dct["__id__"]

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "case_name", # TODO more parameters
    NETWORKS
)
def test_sia(case_name):
    example_func = EXAMPLES["subsystem"][case_name]
    actual = sia(example_func(), parallel=False)
    expected = expected_sia(case_name)
    
    actual = jsonify(actual)
    
    remove_ids(actual)
    remove_ids(expected)
    
    assert actual == expected

# TODO failing via PyTest, but passing in notebook; nested equal dicts flagged not equal
@pytest.mark.parametrize(
    "case_name", # TODO more parameters
    NETWORKS
)
def test_compute_subsystem_ces(case_name):
    example_func = EXAMPLES["subsystem"][case_name]
    actual = ces(example_func())
    expected = expected_ces(case_name)
    
    actual = jsonify(actual)
    
    remove_ids(actual)
    remove_ids(expected)
    
    assert actual == expected

# TODO failing via PyTest, but passing in notebook
@pytest.mark.parametrize(
    "case_name",  # TODO more parameters
    NETWORKS
)
def test_relations(case_name):  # TODO more descriptive name
    subsystem = EXAMPLES["subsystem"][case_name]()
    ces_obj = ces(subsystem)
    actual = relations(subsystem, ces_obj, parallel=False)
    expected = expected_relations(case_name)
    
    actual = jsonify(actual)
    
    for item in actual + expected:
        if isinstance(item, dict):
            remove_ids(item)
    
    assert actual == expected
