#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


collect_ignore = ["setup.py", ".pythonrc.py", "__cyphi_cache__",
                  "test/__cyphi_cache__", "results"]


# Run slow tests separately with command-line option
# ==================================================

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
