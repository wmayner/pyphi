"""Tests that result objects carry a frozen ConfigSnapshot."""

from __future__ import annotations

import pytest

from pyphi import examples
from pyphi.conf import config
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.formalism.queries import sia
from pyphi.system import System


@pytest.fixture
def cs():
    return System(
        substrate=examples.basic_substrate(),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )


class TestSIASnapshot:
    def test_sia_has_config_snapshot(self, cs):
        result = sia(cs)
        assert isinstance(result.config, ConfigSnapshot)

    def test_snapshot_records_precision_at_construction(self, cs):
        with config.override(precision=7):
            result = sia(cs)
        assert result.config.numerics.precision == 7

    def test_mutating_global_after_construction_doesnt_change_snapshot(self, cs):
        result = sia(cs)
        original = result.config.numerics.precision
        try:
            config.precision = 99
            assert result.config.numerics.precision == original
        finally:
            config.precision = 13

    def test_snapshot_records_formalism_at_construction(self, cs):
        result = sia(cs)
        assert result.config.formalism.iit.version == config.formalism.iit.version

    def test_snapshot_as_kwargs_can_reproduce_override(self, cs):
        with config.override(precision=11):
            result = sia(cs)
        kwargs = result.config.as_kwargs()
        assert kwargs["precision"] == 11
        # Re-applying the snapshot's kwargs reproduces the recorded state.
        with config.override(**kwargs):
            assert config.numerics.precision == 11
