"""Behavior tests for the global ``pyphi.config`` singleton.

These cover validation, the override context manager / decorator on the
real layered config, and the logging-callback wiring. The descriptor /
``Option`` infrastructure that previously powered the global was deleted
in P10b, so only the integration-level tests remain.
"""

import logging
from pathlib import Path

import pytest

from pyphi import config


def test_reconfigure_logging_on_change(capsys):
    log = logging.getLogger("pyphi.config")

    with config.override(LOG_STDOUT_LEVEL="WARNING"):
        log.warning("Just a warning, folks.")
    _out, err = capsys.readouterr()
    assert "Just a warning, folks." in err

    with config.override(LOG_STDOUT_LEVEL="ERROR"):
        log.warning("Another warning.")
    _out, err = capsys.readouterr()
    assert err == ""


@config.override()
@pytest.mark.parametrize(
    "name,valid,invalid",
    [
        ("REPR_VERBOSITY", [0, 1, 2], [-1, 3]),
        ("PARALLEL", [True, False], ["True", "False", "no", 0, 1]),
        ("LOG_FILE", ["filename", Path("filename")], [0, 1]),
    ],
)
def test_config_validation(name, valid, invalid):
    for value in valid:
        setattr(config, name, value)

    for value in invalid:
        with pytest.raises(ValueError):
            setattr(config, name, value)
