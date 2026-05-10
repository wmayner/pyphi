"""Tests for _GlobalConfig.install_snapshot."""

from __future__ import annotations

from pyphi.conf import config


def test_install_snapshot_replaces_global_layers():
    original = config.snapshot()
    try:
        with config.override(precision=11, repertoire_measure="L1"):
            captured = config.snapshot()

        assert config.numerics.precision == original.numerics.precision

        config.install_snapshot(captured)
        assert config.numerics.precision == 11
        assert config.formalism.iit.repertoire_measure == "L1"
    finally:
        config.install_snapshot(original)


def test_install_snapshot_idempotent():
    snap = config.snapshot()
    before = config.snapshot()
    config.install_snapshot(snap)
    after = config.snapshot()
    assert before == after
