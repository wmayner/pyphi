#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import test.example_networks


collect_ignore = ["setup.py", ".pythonrc.py"]


@pytest.fixture()
def m():
    return test.example_networks.m()


@pytest.fixture()
def s():
    return test.example_networks.s()
