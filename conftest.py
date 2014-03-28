#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import test


collect_ignore = ["setup.py", ".pythonrc.py"]


@pytest.fixture()
def m():
    return test.m()


@pytest.fixture()
def s():
    return test.s()
