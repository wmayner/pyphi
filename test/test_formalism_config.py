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

    def test_formalism_config_view_updates_on_global_change(self):
        formalism = FORMALISM_REGISTRY["IIT_4_0_2023"]
        instance = formalism() if isinstance(formalism, type) else formalism
        original = instance.config.repertoire_distance
        with config.override(repertoire_distance="EMD"):
            # The view is live; reads reflect the override.
            assert instance.config.repertoire_distance == "EMD"
        assert instance.config.repertoire_distance == original
