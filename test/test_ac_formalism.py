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
