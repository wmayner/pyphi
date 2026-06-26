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
            instance.config.iit.mechanism_phi_measure
            == config.formalism.iit.mechanism_phi_measure
        )

    def test_formalism_config_is_frozen_at_construction(self):
        """The formalism's ``config`` field is captured at instance construction,
        not a live view over the global. This is what lets workers receive the
        formalism with its config attached via cloudpickle (no global-state
        pickling)."""
        formalism = FORMALISM_REGISTRY["IIT_4_0_2023"]
        instance = formalism() if isinstance(formalism, type) else formalism
        captured = instance.config.iit.mechanism_phi_measure
        # EMD is an IIT 3.0 measure; this only checks config-snapshot freezing,
        # not formalism validity, so bypass the B13 combination validator.
        with config.override(mechanism_phi_measure="EMD", validate_config=False):
            # Frozen field; the captured value does NOT track global changes.
            assert instance.config.iit.mechanism_phi_measure == captured
        assert instance.config.iit.mechanism_phi_measure == captured
