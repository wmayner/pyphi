#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conftest.py

import logging
import os
import shutil

import pytest

import example_networks
from pyphi import config, constants, db

log = logging.getLogger()

# Cache management and fixtures
# =============================

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


@pytest.fixture
def flushcache():
    '''Flush the currently enabled cache.'''
    def cache_flusher():
        log.info("FLUSHING CACHE!")
        if config.CACHING_BACKEND == constants.DATABASE:
            _flush_database_cache()
        elif config.CACHING_BACKEND == constants.FILESYSTEM:
            _flush_joblib_cache()
    return cache_flusher


@pytest.fixture(scope="session")
def restore_fs_cache(request):
    '''Temporarily backup, then restore, the user's joblib cache after each
    testing session.'''
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


# Test fixtures from example networks
# =============================================================================


# Matlab standard network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def standard():
    return example_networks.standard()


@pytest.fixture()
def s():
    return example_networks.s()


@pytest.fixture()
def s_empty():
    return example_networks.s_empty()


@pytest.fixture()
def s_single():
    return example_networks.s_single()


@pytest.fixture()
def subsys_n0n2():
    return example_networks.subsys_n0n2()


@pytest.fixture()
def subsys_n1n2():
    return example_networks.subsys_n1n2()


# Noised standard example and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.fixture()
def noised():
    return example_networks.noised()


@pytest.fixture()
def s_noised():
    return example_networks.s_noised()


@pytest.fixture()
def noisy_selfloop_single():
    return example_networks.noisy_selfloop_single()


# Simple network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def simple():
    return example_networks.simple()


@pytest.fixture()
def simple_subsys_all_off():
    return example_networks.simple_subsys_all_off()


@pytest.fixture()
def simple_subsys_all_a_just_on():
    return example_networks.simple_subsys_all_a_just_on()


# Big network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def big():
    return example_networks.big()


@pytest.fixture()
def big_subsys_all():
    return example_networks.big_subsys_all()


@pytest.fixture()
def big_subsys_0_thru_3():
    return example_networks.big_subsys_0_thru_3()


# Trivially reducible network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def reducible():
    return example_networks.reducible()


@pytest.fixture()
def rule152():
    return example_networks.rule152()


@pytest.fixture()
def rule152_s():
    return example_networks.rule152_s()


# Subsystems with complete graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def s_complete():
    return example_networks.s_complete()


@pytest.fixture()
def s_noised_complete():
    return example_networks.s_noised_complete()


@pytest.fixture()
def big_subsys_all_complete():
    return example_networks.big_subsys_all_complete()


@pytest.fixture()
def rule152_s_complete():
    return example_networks.rule152_s_complete()


@pytest.fixture()
def eights_complete():
    return example_networks.eights_complete()


# Macro/Micro networks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def macro():
    return example_networks.macro()


@pytest.fixture()
def macro_s():
    return example_networks.macro_s()


@pytest.fixture()
def micro():
    return example_networks.micro()


@pytest.fixture()
def micro_s():
    return example_networks.micro_s()


@pytest.fixture()
def micro_s_all_off():
    return example_networks.micro_s_all_off()


@pytest.fixture()
def propagation_delay():
    return example_networks.propagation_delay()
