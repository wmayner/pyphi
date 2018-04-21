#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shutil

import pytest

import pyphi
from pyphi import config, constants, db


log = logging.getLogger('pyphi.test')

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


# PyPhi configuration management
# ================================================================

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


# Cache management and fixtures
# ================================================================

# Use a test database if database caching is enabled.
if config.CACHING_BACKEND == constants.DATABASE:
    db.collection = db.database.test

# Backup location for the existing joblib cache directory.
BACKUP_CACHE_DIR = config.FS_CACHE_DIRECTORY + '.BACKUP'


def _flush_joblib_cache():
    # Remove the old joblib cache directory.
    shutil.rmtree(config.FS_CACHE_DIRECTORY)
    # Make a new, empty one.
    os.mkdir(config.FS_CACHE_DIRECTORY)


def _flush_database_cache():
    # Flush the `test` collection in the database.
    return db.database.test.remove({})


# TODO: flush Redis cache
@pytest.fixture(scope="function", autouse=True)
def flushcache(request):
    """Flush the currently enabled cache.

    This is called before every test case.
    """
    log.info("Flushing cache...")
    if config.CACHING_BACKEND == constants.DATABASE:
        _flush_database_cache()
    elif config.CACHING_BACKEND == constants.FILESYSTEM:
        _flush_joblib_cache()


@pytest.fixture(scope="session", autouse=True)
def restore_filesystem_cache(request):
    """Temporarily backup, then restore, the user's joblib cache after each
    testing session.

    This is called before flushcache is, ensuring the cache is saved.
    """
    # Move the joblib cache to a backup location and create a fresh cache if
    # filesystem caching is enabled
    if config.CACHING_BACKEND == constants.FILESYSTEM:
        if os.path.exists(BACKUP_CACHE_DIR):
            raise Exception("You must move the backup of the filesystem cache "
                            "at " + BACKUP_CACHE_DIR + " before running the "
                            "test suite.")
        shutil.move(config.FS_CACHE_DIRECTORY, BACKUP_CACHE_DIR)
        os.mkdir(config.FS_CACHE_DIRECTORY)

    def fin():
        if config.CACHING_BACKEND == constants.FILESYSTEM:
            # Remove the tests' joblib cache directory.
            shutil.rmtree(config.FS_CACHE_DIRECTORY)
            # Restore the old joblib cache.
            shutil.move(BACKUP_CACHE_DIR,
                        config.FS_CACHE_DIRECTORY)

    # Restore the cache after the last test with this fixture has run
    request.addfinalizer(fin)
