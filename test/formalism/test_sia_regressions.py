"""Regression tests for IIT 4.0 SIA edge cases.

Covers single-unit systems under ``DIRECTED_BIPARTITION_CUT_ONE``, the
Eq. 14 disconnection filter for ``EDGE_CUT_BIDIRECTIONAL``, single-direction
analyses with tied specified states, the membership of ``sia.ties``, and the
determinism of the mechanism MIP under ``shortcircuit_sia=False``.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyphi
from pyphi import connectivity
from pyphi import examples
from pyphi.conf import presets
from pyphi.direction import Direction


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


# ---- single-unit systems under DIRECTED_BIPARTITION_CUT_ONE ----


def test_monad_sia_under_cut_one_scheme_is_zero():
    """A single unit has no valid bipartition-of-one, so φ_s is 0 (as under
    every other scheme), not a ZeroDivisionError."""
    substrate = pyphi.Substrate(
        np.array([[0.0], [1.0]]), cm=np.array([[1]]), node_labels=["A"]
    )
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(system_partition_scheme="DIRECTED_BIPARTITION_CUT_ONE"),
    ):
        sia = pyphi.formalism.sia(pyphi.System(substrate, (1,), (0,)))
    assert float(sia.phi) == 0.0


def test_single_unit_subset_sia_under_cut_one_scheme_is_zero():
    substrate = examples.basic_noisy_selfloop_substrate()
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(system_partition_scheme="DIRECTED_BIPARTITION_CUT_ONE"),
    ):
        result = pyphi.analyze(substrate, (1, 1, 0), subset=(0,), compute="sia")
    assert float(result.phi) == 0.0


# ---- Eq. 14 disconnection filter for edge-cut schemes ----


def test_edge_cut_bidirectional_mip_disconnects_the_system():
    """The MIP under EDGE_CUT_BIDIRECTIONAL must be a disconnecting cut
    (Eq. 14); grid3 is irreducible, so φ_s is positive."""
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(system_partition_scheme="EDGE_CUT_BIDIRECTIONAL"),
    ):
        system = examples.grid3_system()
        sia = system.sia()
        assert not connectivity.is_strong(system.apply_cut(sia.partition).proper_cm)
    assert float(sia.phi) == pytest.approx(0.063699, abs=1e-5)


def test_edge_cut_all_filter_still_applies():
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(system_partition_scheme="EDGE_CUT_ALL"),
    ):
        system = examples.grid3_system()
        sia = system.sia()
        assert not connectivity.is_strong(system.apply_cut(sia.partition).proper_cm)
    assert float(sia.phi) == pytest.approx(0.024666, abs=1e-5)


# ---- single-direction analysis with a tied specified state ----


@pytest.mark.parametrize(
    "preset_name,expected",
    [("iit4_2023", 1.5), ("iit4_2026", 1.0)],
)
def test_single_direction_sia_resolves_ties_within_that_direction(preset_name, expected):
    """xor at (0, 0, 0) has tied specified cause states; a CAUSE-only
    analysis must resolve the tie within the cause direction rather than
    raising on the empty effect-candidate set."""
    preset = getattr(presets, preset_name)
    with pyphi.config.override(**preset):
        system = pyphi.System(examples.xor_substrate(), (0, 0, 0), (0, 1, 2))
        sia = system.sia(directions=[Direction.CAUSE])
    assert float(sia.phi) == pytest.approx(expected)


def test_single_direction_sia_is_deterministic():
    with pyphi.config.override(**presets.iit4_2026):
        system = pyphi.System(examples.xor_substrate(), (0, 0, 0), (0, 1, 2))
        first = system.sia(directions=[Direction.CAUSE])
        system.clear_caches()
        second = system.sia(directions=[Direction.CAUSE])
    assert float(first.phi) == float(second.phi)
    assert first.system_state.cause.state == second.system_state.cause.state


# ---- sia.ties contains only genuinely tied readings ----


@pytest.fixture
def asymmetric_tied_state_substrate():
    """Substrate whose specified cause state ties at (1, 0, 1) but whose
    per-reading φ_s values differ, so the cascade resolves the tie at the
    Integration level."""
    tpm = np.array(
        [
            [0.9, 0.25, 0.9],
            [0.5, 0.5, 0.75],
            [0.75, 0.9, 0.9],
            [0.1, 0.5, 0.9],
            [0.25, 0.1, 0.5],
            [0.75, 0.75, 0.9],
            [0.75, 0.1, 0.9],
            [0.5, 0.9, 0.9],
        ]
    )
    return pyphi.Substrate(tpm, node_labels=list("ABC"))


def test_sia_ties_exclude_readings_that_lost_the_cascade(
    asymmetric_tied_state_substrate,
):
    """Specified-state readings whose φ_s lost the cascade are not ties: every
    member of ``sia.ties`` must be φ-equal to the winner."""
    from pyphi import numerics

    with pyphi.config.override(**presets.iit4_2026):
        sia = pyphi.System(asymmetric_tied_state_substrate, (1, 0, 1)).sia()
    assert len(sia.ties) == 1
    for tied in sia.ties:
        assert numerics.eq(float(tied.phi), float(sia.phi))
    # The tie metadata still surfaces the full tied specified-state set.
    assert len(sia.system_state.cause.ties) == 2


def test_sia_ties_keep_genuinely_tied_readings():
    """xor at (0, 0, 0): both tied cause readings have equal φ_s and both
    remain in ``sia.ties``."""
    with pyphi.config.override(**presets.iit4_2026):
        sia = pyphi.System(examples.xor_substrate(), (0, 0, 0)).sia()
    assert len(sia.ties) == 2
    phis = {round(float(t.phi), 12) for t in sia.ties}
    assert len(phis) == 1


# ---- mechanism MIP determinism under shortcircuit_sia=False ----


def test_never_shortcircuit_sentinel_preserves_enumeration_order():
    """The never-short-circuit sentinel must not switch parallel collection
    to completion order: MIP tie resolution depends on enumeration order."""
    import time

    from pyphi.formalism.queries import _never_shortcircuit
    from pyphi.parallel import map_reduce

    def slow_identity(i):
        time.sleep((10 - i) * 0.01)
        return i

    out = map_reduce(
        slow_identity,
        list(range(10)),
        parallel=True,
        backend="thread",
        sequential_threshold=1,
        chunksize=1,
        shortcircuit_func=_never_shortcircuit,
        progress=False,
    )
    assert out == list(range(10))


def test_mechanism_mip_deterministic_without_shortcircuit():
    """With ``shortcircuit_sia=False`` and parallel partition evaluation, the
    reported mechanism MIP must match the sequential result on every run."""
    from pyphi.formalism import queries
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(**presets.iit4_2026):
        system = examples.xor_system()
        mechanism_measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.mechanism_phi_measure
        )
        specification_measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.specification_measure
        )

        def run_mice(**overrides):
            with pyphi.config.override(
                progress_bars=False, shortcircuit_sia=False, **overrides
            ):
                return queries.find_mice(
                    system,
                    Direction.CAUSE,
                    (0, 1, 2),
                    mechanism_measure=mechanism_measure,
                    specification_measure=specification_measure,
                )

        sequential = run_mice(parallel=False)
        forced = dict(
            pyphi.config.infrastructure.parallel_mechanism_partition_evaluation
        )
        forced.update(parallel=True, sequential_threshold=1, chunksize=1, progress=False)
        for _ in range(5):
            parallel_result = run_mice(
                parallel=True,
                parallel_backend="thread",
                parallel_mechanism_partition_evaluation=forced,
            )
            assert parallel_result.partition == sequential.partition
            assert float(parallel_result.phi) == pytest.approx(float(sequential.phi))
