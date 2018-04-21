#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytest

import pyphi

collect_ignore = [
    "setup.py",
    ".pythonrc.py"
]
# Also ignore everything that git ignores.
git_ignore = os.path.join(os.path.dirname(__file__), '.gitignore')
collect_ignore += list(filter(None, open(git_ignore).read().split('\n')))

# Run slow tests separately with command-line option, filter tests
# ================================================================


def pytest_addoption(parser):
    parser.addoption("--filter", action="store",
                     help="only run tests with the given mark")
    parser.addoption("--slow", action="store_true", help="run slow tests")
    parser.addoption("--veryslow", action="store_true",
                     help="run very slow tests")


def pytest_runtest_setup(item):
    filt = item.config.getoption("--filter")
    if filt:
        if filt not in item.keywords:
            pytest.skip("only running tests with the '{}' mark".format(filt))
    else:
        if 'slow' in item.keywords and not item.config.getoption("--slow"):
            pytest.skip("need --slow option to run")
        if ('veryslow' in item.keywords and
                not item.config.getoption("--veryslow")):
            pytest.skip("need --veryslow option to run")


@pytest.fixture(scope='function')
def restore_config_afterwards():
    '''Reset PyPhi configuration after a test.

    Useful for doctests that can't be decorated with `config.override`.
    '''
    with pyphi.config.override():
        yield


@pytest.fixture(scope='session', autouse=True)
def disable_progress_bars():
    '''Disable progress bars during tests.

    Progress bars are already disabled for unit tests; doctests work
    differently, I think because of output redirection.
    '''
    with pyphi.config.override(PROGRESS_BARS=False):
        yield
