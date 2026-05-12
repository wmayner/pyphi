#!/usr/bin/env python3

import logging
from pathlib import Path

import pytest
import yaml

import pyphi

log = logging.getLogger("pyphi.test")

collect_ignore = ["setup.py", ".pythonrc.py"]
# Also ignore everything that git ignores.
with open(Path(__file__).parent / ".gitignore") as f:
    collect_ignore += list(filter(None, f.read().split("\n")))


IIT_3_CONFIG = "pyphi_config_3.0.yml"

# Run slow tests separately with command-line option, filter tests
# ================================================================


def pytest_addoption(parser):
    parser.addoption(
        "--filter", action="store", help="only run tests with the given mark"
    )
    parser.addoption("--outdated", action="store_true", help="run outdated tests")
    parser.addoption("--slow", action="store_true", help="run slow tests")
    parser.addoption("--veryslow", action="store_true", help="run very slow tests")


def pytest_runtest_setup(item):
    filt = item.config.getoption("--filter")
    if filt:
        if filt not in item.keywords:
            pytest.skip(f"only running tests with the '{filt}' mark")
    else:
        if "outdated" in item.keywords and not item.config.getoption("--outdated"):
            pytest.skip("need --outdated option to run")
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


@pytest.fixture(autouse=True)
def _restore_config_after_test():
    """Snapshot the global config before each test and restore after.

    Defensive test hygiene against any test or doctest that mutates
    ``pyphi.config`` without unwinding (e.g. a raw assignment outside an
    ``override`` block). Lives at the root so it covers doctests collected
    from ``pyphi/`` as well as tests under ``test/``.
    """
    snapshot = pyphi.config.snapshot()
    try:
        yield
    finally:
        pyphi.config.install_snapshot(snapshot)


@pytest.fixture(scope="function")
def use_iit_3_config():
    """Use the IIT-3 configuration for all tests."""
    with open(IIT_3_CONFIG) as f:
        iit3_config = yaml.load(f, Loader=yaml.SafeLoader)
    with pyphi.config.override(**iit3_config):
        yield


# Cache management and fixtures
# ================================================================


@pytest.fixture(scope="function", autouse=True)
def flushcache(request):  # noqa: ARG001
    """No-op cache flush between tests.

    PyPhi's caches are designed to be safe to share across tests:
    combinatorial caches in ``partition.py`` / ``distribution.py`` /
    ``combinatorics.py`` memoize pure functions (no per-test state to
    pollute); the kernel ``_memoize`` keys on ``id(cs)`` and uses
    ``weakref.finalize`` to evict cache entries when a CandidateSystem
    is garbage-collected; Network purview caches are anonymous and die
    with their Network. Clearing them between every test forces
    expensive re-enumeration of partitions on every fixture setup
    (5x suite slowdown observed during P9 bisect).
    """
    log.info("Flushing caches... (no-op)")


# Parallel (local backend)
# ================================================================


@pytest.fixture(scope="module")
def parallel_context():
    """Set up parallel computation context.

    With the local backend, no special initialization is needed.
    This fixture is kept for API compatibility with existing tests.
    """
    # Local backend uses ProcessPoolExecutor which doesn't require init
    yield None
