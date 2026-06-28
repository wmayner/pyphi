"""Behavior tests for the global ``pyphi.config`` singleton.

These cover validation and the override context manager / decorator on the
real layered config. The descriptor / ``Option`` infrastructure that
previously powered the global was deleted in P10b, so only the
integration-level tests remain.
"""

import pytest

from pyphi import config


@config.override()
@pytest.mark.parametrize(
    "name,valid,invalid",
    [
        ("REPR_VERBOSITY", [0, 1, 2, 3, 4], [-1, 5]),
        ("PARALLEL", [True, False], ["True", "False", "no", 0, 1]),
    ],
)
def test_config_validation(name, valid, invalid):
    for value in valid:
        setattr(config, name, value)

    for value in invalid:
        with pytest.raises(ValueError):
            setattr(config, name, value)
