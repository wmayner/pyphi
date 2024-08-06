#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import pytest
import ray
import yaml

import pyphi
import pyphi.cache.redis as redis
from pyphi.conf import config

log = logging.getLogger("pyphi.test")

collect_ignore = ["setup.py", ".pythonrc.py"]
# Also ignore everything that git ignores.
with open(Path(__file__).parent / ".gitignore", mode="rt") as f:
    collect_ignore += list(filter(None, f.read().split("\n")))


IIT_3_CONFIG = "pyphi_config_3.0.yml"

# Run slow tests separately with command-line option, filter tests
# ================================================================


def pytest_addoption(parser):
    parser.addoption(
        "--filter", action="store", help="only run tests with the given mark"
    )
    parser.addoption("--slow", action="store_true", help="run slow tests")
    parser.addoption("--veryslow", action="store_true", help="run very slow tests")


def pytest_runtest_setup(item):
    filt = item.config.getoption("--filter")
    if filt:
        if filt not in item.keywords:
            pytest.skip("only running tests with the '{}' mark".format(filt))
    else:
        if "slow" in item.keywords and not item.config.getoption("--slow"):
            pytest.skip("need --slow option to run")
        if "veryslow" in item.keywords and not item.config.getoption("--veryslow"):
            pytest.skip("need --veryslow option to run")


# PyPhi configuration management
# ================================================================


@pytest.fixture(scope="function")
def restore_config_afterwards():
    """Reset PyPhi configuration after a test.

    Useful for doctests that can't be decorated with `config.override`.
    """
    with pyphi.config.override():
        yield


@pytest.fixture(scope="session", autouse=True)
def disable_progress_bars():
    """Disable progress bars during tests.

    Without this progress bars are already disabled for unit tests; doctests
    work differently, I think because of output redirection.
    """
    with pyphi.config.override(PROGRESS_BARS=False):
        yield


@pytest.fixture(scope="function")
def use_iit_3_config():
    """Use the IIT-3 configuration for all tests."""
    with open(IIT_3_CONFIG, mode="rt") as f:
        iit3_config = yaml.load(f, Loader=yaml.SafeLoader)
    with pyphi.config.override(**iit3_config):
        yield


# Cache management and fixtures
# ================================================================


@pytest.fixture(scope="session", autouse=True)
def protect_caches(request):
    """Temporarily backup, then restore, the user's Redis caches
    before and after the testing session.

    This is called before flushcache, ensuring the cache is saved.
    """
    # Initialize a test Redis connection
    original_redis_conn = redis.conn
    redis.conn = redis.init(config.REDIS_CONFIG["test_db"])
    yield
    # Restore the cache after the last test has run
    redis.conn = original_redis_conn


def _flush_redis_cache():
    if redis.available():
        redis.conn.flushdb()
        redis.conn.config_resetstat()


# TODO: flush Redis cache
@pytest.fixture(scope="function", autouse=True)
def flushcache(request):
    """Flush the currently enabled cache.

    This is called before every test case.
    """
    log.info("Flushing caches...")
    _flush_redis_cache()


# Ray
# ================================================================


@pytest.fixture(scope="module")
def ray_context():
    context = ray.init(num_cpus=3)
    yield context
    ray.shutdown()
