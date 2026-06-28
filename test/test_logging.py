"""Silent-by-default logging and the enable_logging opt-in."""

from __future__ import annotations

import logging

import pytest

from pyphi import enable_logging
from pyphi import log as plog


def _pyphi_logger() -> logging.Logger:
    return logging.getLogger("pyphi")


def _installed_handlers() -> list[logging.Handler]:
    """PyPhi-logger handlers that are not the import-time NullHandler."""
    return [
        h for h in _pyphi_logger().handlers if not isinstance(h, logging.NullHandler)
    ]


@pytest.fixture
def restore_pyphi_logging():
    """Snapshot and restore the pyphi logger so a test cannot leak handlers."""
    logger = _pyphi_logger()
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    for handler in logger.handlers:
        if handler not in saved[0]:
            handler.close()
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_null_handler_present_by_default():
    assert any(isinstance(h, logging.NullHandler) for h in _pyphi_logger().handlers)


def test_enable_logging_attaches_one_handler_at_info(restore_pyphi_logging):
    enable_logging()
    installed = _installed_handlers()
    assert len(installed) == 1
    assert isinstance(installed[0], plog.TqdmHandler)
    assert _pyphi_logger().level == logging.INFO
    assert _pyphi_logger().propagate is False


def test_enable_logging_replaces_rather_than_stacks(tmp_path, restore_pyphi_logging):
    enable_logging()
    log_path = tmp_path / "run.log"
    enable_logging(level="DEBUG", file=str(log_path))
    installed = _installed_handlers()
    assert len(installed) == 1
    assert isinstance(installed[0], logging.FileHandler)
    _pyphi_logger().debug("hello-from-test")
    assert "hello-from-test" in log_path.read_text()


def test_enable_logging_rejects_unknown_level(restore_pyphi_logging):
    with pytest.raises(ValueError):
        enable_logging(level="NOPE")


def test_setting_removed_log_field_raises(restore_pyphi_logging):
    from pyphi.conf import config
    from pyphi.conf._field_routing import ConfigurationError

    with pytest.raises(ConfigurationError):
        config.log_file_level = "INFO"


def test_import_writes_no_log_file(tmp_path):
    import os
    import subprocess
    import sys

    env = {"PYPHI_WELCOME_OFF": "1", "PATH": os.environ["PATH"]}
    subprocess.run(
        [sys.executable, "-c", "import pyphi"],
        cwd=tmp_path,
        check=True,
        env=env,
    )
    assert not (tmp_path / "pyphi.log").exists()
