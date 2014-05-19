#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import example_networks


# Test fixtures from example networks
# ===================================

# Matlab standard network and subsystems

@pytest.fixture()
def standard():
    return example_networks.standard()


@pytest.fixture()
def s():
    return example_networks.s()


@pytest.fixture()
def subsys_n0n2():
    return example_networks.subsys_n0n2()


@pytest.fixture()
def subsys_n1n2():
    return example_networks.subsys_n1n2()


# Simple network and subsystems

@pytest.fixture()
def simple():
    return example_networks.simple()


@pytest.fixture()
def s_subsys_all_off():
    return example_networks.s_subsys_all_off()


@pytest.fixture()
def s_subsys_all_a_just_on():
    return example_networks.s_subsys_all_a_just_on()


# Big network and subsystems

@pytest.fixture()
def big():
    return example_networks.big()


@pytest.fixture()
def big_subsys_all():
    return example_networks.big_subsys_all()


# Reducible network

@pytest.fixture()
def reducible():
    return example_networks.reducible()
