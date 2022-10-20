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
from pyphi.relations import relations

NETWORKS = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]

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

def expected_json(type, example):
    PATH = f"test/data/new_big_phi/{type}/{type}_{example}.json"
    
    with open(PATH) as f:
        expected = json.load(f)
    
    return expected

def assert_equality(actual, expected):
    actual = jsonify(actual)
    
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        
        remove_ids(actual)
        remove_ids(expected)
        
    else:  # should be lists if not dictionaries
        assert isinstance(actual, list)
        assert isinstance(expected, list)
        
        for item in actual + expected:
            if isinstance(item, dict):
                remove_ids(item)
    
    assert actual == expected

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize(
    "case_name",
    NETWORKS
)
def test_sia(case_name):
    example_func = EXAMPLES["subsystem"][case_name]
    actual = sia(example_func(), parallel=False)
    expected = expected_json("sia", case_name)
    
    assert_equality(actual, expected)

@pytest.mark.parametrize(
    "case_name",
    NETWORKS
)
def test_compute_subsystem_ces(case_name):
    example_func = EXAMPLES["subsystem"][case_name]
    actual = ces(example_func())
    expected = expected_json("ces", case_name)
    
    assert_equality(actual, expected)

@pytest.mark.parametrize(
    "case_name",
    NETWORKS
)
def test_relations(case_name):
    subsystem = EXAMPLES["subsystem"][case_name]()
    ces_obj = ces(subsystem)
    actual = relations(subsystem, ces_obj, parallel=False)
    expected = expected_json("relations", case_name)
    
    assert_equality(actual, expected)
