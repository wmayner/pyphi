#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


collect_ignore = [
    "setup.py",
    ".pythonrc.py",
    "__cyphi_cache__",
    "__cyphi_cache__.BACKUP",
    "test/__cyphi_cache__",
    "results",
    "build"
]


# Run slow tests separately with command-line option
# ==================================================

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")
    parser.addoption("--veryslow", action="store_true",
                     help="run very slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("need --slow option to run")
    if 'veryslow' in item.keywords and not item.config.getoption("--veryslow"):
        pytest.skip("need --veryslow option to run")
