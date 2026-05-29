"""Tests for the Actual Causation formalism object and registry."""

from __future__ import annotations

import pytest


def test_ac_formalism_registry_exists_and_is_typed():
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY
    from pyphi.formalism.base import ActualCausationFormalismRegistry

    assert isinstance(
        ACTUAL_CAUSATION_FORMALISM_REGISTRY, ActualCausationFormalismRegistry
    )


def test_ac_formalism_registry_rejects_wrong_shape():
    from pyphi.formalism.base import ActualCausationFormalismRegistry

    registry = ActualCausationFormalismRegistry()
    with pytest.raises(TypeError):
        registry.register("BOGUS", object())


def test_ac_config_version_default_and_override():
    from pyphi import config

    assert config.formalism.actual_causation.version == "AC_2019"
    with config.override({"actual_causation.version": "AC_2019"}):
        assert config.formalism.actual_causation.version == "AC_2019"


def test_version_is_colliding_field():
    """``version`` exists in both IIT and AC sub-namespaces, so the bare leaf
    is ambiguous: flat reads/writes raise and the dotted forms are required."""
    from pyphi import config
    from pyphi.conf._field_routing import ConfigurationError
    from pyphi.conf._field_routing import colliding_formalism_fields

    assert "version" in colliding_formalism_fields()
    with pytest.raises(AttributeError):
        _ = config.version
    with pytest.raises(ConfigurationError), config.override(version="AC_2019"):
        pass
