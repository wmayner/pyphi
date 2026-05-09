"""Tests for PhiFormalism + FormalismConfig integration."""

from __future__ import annotations

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism import FORMALISM_REGISTRY
from pyphi.formalism.base import PhiFormalism


class TestPhiFormalismHasConfig:
    def test_iit3_formalism_satisfies_protocol(self):
        formalism = FORMALISM_REGISTRY["IIT_3_0"]
        instance = formalism() if isinstance(formalism, type) else formalism
        assert isinstance(instance, PhiFormalism)
        assert isinstance(instance.config, FormalismConfig)

    def test_iit4_2023_formalism_satisfies_protocol(self):
        formalism = FORMALISM_REGISTRY["IIT_4_0_2023"]
        instance = formalism() if isinstance(formalism, type) else formalism
        assert isinstance(instance, PhiFormalism)
        assert isinstance(instance.config, FormalismConfig)

    def test_iit4_2026_formalism_satisfies_protocol(self):
        formalism = FORMALISM_REGISTRY["IIT_4_0_2026"]
        instance = formalism() if isinstance(formalism, type) else formalism
        assert isinstance(instance, PhiFormalism)
        assert isinstance(instance.config, FormalismConfig)

    def test_formalism_config_reflects_global(self):
        formalism = FORMALISM_REGISTRY["IIT_4_0_2023"]
        instance = formalism() if isinstance(formalism, type) else formalism
        assert (
            instance.config.repertoire_distance == config.formalism.repertoire_distance
        )

    def test_formalism_config_is_frozen_at_construction(self):
        """Phase 1 of P11 made formalisms frozen dataclasses — the ``config``
        field is captured at instance construction, not a live view over the
        global. This is what lets workers receive the formalism with its
        config attached via cloudpickle (no global-state pickling)."""
        formalism = FORMALISM_REGISTRY["IIT_4_0_2023"]
        instance = formalism() if isinstance(formalism, type) else formalism
        captured = instance.config.repertoire_distance
        with config.override(repertoire_distance="EMD"):
            # Frozen field; the captured value does NOT track global changes.
            assert instance.config.repertoire_distance == captured
        assert instance.config.repertoire_distance == captured
