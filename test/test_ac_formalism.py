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


def test_ac2019_formalism_registered_and_satisfies_protocol():
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY
    from pyphi.formalism.base import ActualCausationFormalism

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    assert isinstance(formalism, ActualCausationFormalism)
    assert formalism.name == "AC_2019"
    assert "PMI" in formalism.compatible_measures
    assert "WPMI" in formalism.compatible_measures


def test_ac_formalism_rejects_incompatible_measure():
    from pyphi.formalism.actual_causation.formalism import _resolve_ac_measures
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY
    from pyphi.formalism.base import MeasureNotCompatibleError

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    # An IIT measure name is not in AC's compatible_measures.
    with pytest.raises(MeasureNotCompatibleError):
        _resolve_ac_measures(
            formalism, alpha_measure_name="GENERALIZED_INTRINSIC_DIFFERENCE"
        )


def test_ac_formalism_resolves_default_measures():
    from pyphi.formalism.actual_causation.formalism import _resolve_ac_measures
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    resolved = _resolve_ac_measures(formalism)
    assert callable(resolved["alpha_measure"])
    assert callable(resolved["partitioned_repertoire_scheme"])


def test_ac_formalism_unknown_version_raises():
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    with pytest.raises(KeyError):
        ACTUAL_CAUSATION_FORMALISM_REGISTRY["NOPE"]


def test_public_api_dispatches_through_active_formalism():
    """The public AC entry points resolve the formalism from the registry by
    ``config.formalism.actual_causation.version``; an unknown version raises
    KeyError, which is only possible if dispatch goes through the registry."""
    from pyphi import actual
    from pyphi import config
    from pyphi import examples

    transition = examples.prevention_transition()
    # Default version computes a result.
    assert actual.account(transition) is not None
    with config.override({"actual_causation.version": "NOPE"}):
        with pytest.raises(KeyError):
            actual.account(transition)
        with pytest.raises(KeyError):
            actual.sia(transition)
        with pytest.raises(KeyError):
            transition.find_causal_link(actual.Direction.CAUSE, (2,))
