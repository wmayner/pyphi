"""Tests that result objects carry a frozen ConfigSnapshot."""

from __future__ import annotations

import pytest

from pyphi import actual
from pyphi import examples
from pyphi.conf import config
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.formalism import iit3
from pyphi.formalism.queries import sia
from pyphi.system import System

from .conftest import IIT_3_CONFIG


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

    def test_iit4_sia_recompute_under_recorded_snapshot_matches(self, cs):
        """``install_snapshot(r1.config)`` + recompute yields ``r2 == r1``.

        Closes the round-trip claim for IIT 4.0 SIA at every layer the
        snapshot records (collision-safe via :meth:`install_snapshot`).
        """
        with config.override(precision=11):
            r1 = sia(cs)
        original = config.snapshot()
        try:
            config.install_snapshot(r1.config)
            r2 = sia(cs)
        finally:
            config.install_snapshot(original)
        assert r1 == r2


@pytest.fixture
def transition():
    return examples.prevention_transition()


class TestIIT3SIASnapshot:
    """The IIT 3.0 SIA path must propagate the parent's snapshot to workers.

    Previously, ``MapReduce._run_parallel`` dispatched ``map_func`` to
    loky workers without wrapping it in ``_make_worker_fn``. Workers
    computed under their default config (IIT 4.0) and the result's
    ``.config`` recorded that default instead of the parent's IIT 3.0.
    """

    def test_iit3_sia_records_parent_iit3_version_sequential(self, cs):
        with IIT_3_CONFIG, config.override(parallel=False):
            r = iit3.sia(cs)
        assert r.config.formalism.iit.version == "IIT_3_0", (
            "Sequential IIT 3.0 SIA must record parent's iit.version; "
            f"got {r.config.formalism.iit.version!r}."
        )

    def test_iit3_sia_records_parent_iit3_version_parallel(self, cs):
        with IIT_3_CONFIG, config.override(parallel=True):
            r = iit3.sia(cs)
        assert r.config.formalism.iit.version == "IIT_3_0", (
            "Parallel IIT 3.0 SIA must record parent's iit.version. "
            "If this fails, the legacy MapReduce path isn't wrapping "
            "map_func with _make_worker_fn before dispatching. Got "
            f"{r.config.formalism.iit.version!r}."
        )

    def test_iit3_sia_recompute_under_recorded_snapshot_matches(self, cs):
        with IIT_3_CONFIG, config.override(parallel=False):
            r1 = iit3.sia(cs)
        original = config.snapshot()
        try:
            config.install_snapshot(r1.config)
            r2 = iit3.sia(cs)
        finally:
            config.install_snapshot(original)
        assert r1 == r2


class TestAcSIASnapshot:
    def test_acsia_has_config_snapshot(self, transition):
        with IIT_3_CONFIG:
            result = actual.sia(transition)
        assert isinstance(result.config, ConfigSnapshot), (
            "AcSystemIrreducibilityAnalysis must carry a ConfigSnapshot to "
            "satisfy the reproducibility claim."
        )

    def test_acsia_records_alpha_measure_at_construction(self, transition):
        with IIT_3_CONFIG:
            result = actual.sia(transition)
        assert result.config.formalism.actual_causation.alpha_measure == "PMI"
