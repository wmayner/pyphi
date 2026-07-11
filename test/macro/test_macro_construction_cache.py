"""Tests for the macro-construction intermediate cache."""

import pytest

import pyphi
from pyphi.conf.infrastructure import InfrastructureConfig


class TestConfigOption:
    def test_default_on(self):
        assert InfrastructureConfig().cache_macro_construction is True
        assert pyphi.config.infrastructure.cache_macro_construction is True

    def test_validation_rejects_non_bool(self):
        with pytest.raises(ValueError):
            InfrastructureConfig(cache_macro_construction="yes")

    def test_top_level_override_routes(self):
        with pyphi.config.override(cache_macro_construction=False):
            assert pyphi.config.infrastructure.cache_macro_construction is False
        assert pyphi.config.infrastructure.cache_macro_construction is True
