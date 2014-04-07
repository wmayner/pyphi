#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import test.example_networks


collect_ignore = ["setup.py", ".pythonrc.py"]


# Test fixtures from example networks
# ===================================


@pytest.fixture()
def m():
    return test.example_networks.m()


@pytest.fixture()
def s():
    return test.example_networks.s()


# Run slow tests separately with command-line option
# ==================================================


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
