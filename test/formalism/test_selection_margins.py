"""Tests for selection-margin reporting on the IIT 4.0 SIA."""

import pytest

import pyphi
from pyphi import examples
from pyphi import utils
from pyphi.conf import config
from pyphi.core import repertoire_algebra as ra
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.formalism.iit4 import evaluate_partition
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.partition import system_partitions


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


@pytest.fixture(scope="module")
def basic_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def xor_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.xor_system().sia()


def _per_state_ii(system, direction):
    """Brute force: intrinsic information of every candidate system state."""
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    from pyphi.utils import all_states

    sizes = tuple(alphabet[i] for i in system.node_indices)
    return {
        state: float(
            ra.intrinsic_information(
                system,
                direction,
                mechanism=system.node_indices,
                purview=system.node_indices,
                specification_measure=measure,
                states=[state],
            ).intrinsic_information
        )
        for state in all_states(sizes)
    }


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_state_runner_up_matches_brute_force(basic_sia, direction):
    system = examples.basic_system()
    values = _per_state_ii(system, direction)
    ranked = sorted(values.values(), reverse=True)
    spec = basic_sia.system_state[direction]
    assert float(spec.intrinsic_information) == pytest.approx(ranked[0])
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(ranked[1])
    assert float(spec.state_margin) == pytest.approx(ranked[0] - ranked[1])
    assert spec.runner_up_state in values
    assert values[spec.runner_up_state] == pytest.approx(ranked[1])


def test_state_margin_zero_for_exactly_tied_states(xor_sia):
    # xor at (0, 0, 0): the specified cause state ties exactly (2 tied specs)
    spec = xor_sia.system_state.cause
    assert len(spec.ties) > 1
    assert float(spec.state_margin) == pytest.approx(0.0)
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(
        float(spec.intrinsic_information)
    )


def test_tie_members_share_runner_up_fields(xor_sia):
    specs = xor_sia.system_state.cause.ties
    values = {float(s.runner_up_intrinsic_information) for s in specs}
    assert len(values) == 1


def test_state_margin_none_when_no_competitor():
    system = examples.basic_system()
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    spec = ra.intrinsic_information(
        system,
        Direction.CAUSE,
        mechanism=system.node_indices,
        purview=system.node_indices,
        specification_measure=measure,
        states=[system.proper_state],
    )
    assert spec.runner_up_intrinsic_information is None
    assert spec.runner_up_state is None
    assert spec.state_margin is None


def _brute_force_partition_values(system, system_state):
    measure = resolve_system_measure(config.formalism.iit.system_phi_measure)
    partitions = system_partitions(
        system.node_indices,
        node_labels=system.node_labels,
        partition_scheme=config.formalism.iit.system_partition_scheme,
    )
    return sorted(
        float(
            evaluate_partition(
                partition, system, system_state, system_measure=measure
            ).normalized_phi
        )
        for partition in partitions
    )


def test_partition_margin_matches_brute_force(basic_sia):
    values = _brute_force_partition_values(
        examples.basic_system(), basic_sia.system_state
    )
    assert float(basic_sia.normalized_phi) == pytest.approx(values[0])
    assert float(basic_sia.partition_margin) == pytest.approx(values[1] - values[0])


def test_partition_margin_zero_for_symmetric_substrate():
    # grid3's two best partitions are symmetry-related and tie in
    # normalized phi, so the partition selection is effectively tied.
    sia = examples.grid3_system().sia()
    assert float(sia.partition_margin) == pytest.approx(0.0)
    assert sia.effectively_tied


def test_effectively_tied_fires_on_state_tie(xor_sia):
    assert utils.eq(float(xor_sia.state_margins[Direction.CAUSE]), 0.0)
    assert xor_sia.effectively_tied


def test_untied_system_is_not_flagged(basic_sia):
    assert basic_sia.partition_margin is not None
    assert not utils.eq(float(basic_sia.partition_margin), 0.0)
    assert all(
        margin is None or not utils.eq(float(margin), 0.0)
        for margin in basic_sia.state_margins.values()
    )
    assert not basic_sia.effectively_tied


def test_state_margins_read_through_system_state(basic_sia):
    for direction in Direction.both():
        expected = basic_sia.system_state[direction].state_margin
        assert float(basic_sia.state_margins[direction]) == pytest.approx(
            float(expected)
        )


def test_null_sia_has_no_margins():
    sia = NullSystemIrreducibilityAnalysis()
    assert sia.partition_margin is None
    assert sia.state_margins == {
        Direction.CAUSE: None,
        Direction.EFFECT: None,
    }
    assert not sia.effectively_tied


def test_2026_cap_does_not_change_margins():
    # The 2026 formalism selects the MIP exactly as 2023 does and applies
    # the ii(s) cap afterwards, so both selection margins are identical.
    with pyphi.config.override(**{"iit.version": "IIT_4_0_2026"}):
        sia_2026 = examples.basic_system().sia()
    with pyphi.config.override(**{"iit.version": "IIT_4_0_2023"}):
        sia_2023 = examples.basic_system().sia()
    assert float(sia_2026.partition_margin) == pytest.approx(
        float(sia_2023.partition_margin)
    )
    for direction in Direction.both():
        assert float(sia_2026.state_margins[direction]) == pytest.approx(
            float(sia_2023.state_margins[direction])
        )


def test_runner_up_surface_unchanged(basic_sia):
    # The existing runner-up record keeps its raw-phi semantics.
    assert basic_sia.runner_up is not None
    assert float(basic_sia.runner_up.phi) > float(basic_sia.phi)


def _findings_by_kind(explanation):
    by_kind: dict[str, list] = {}
    for finding in explanation.findings:
        by_kind.setdefault(finding.kind, []).append(finding)
    return by_kind


def test_explain_reports_margins(basic_sia):
    by_kind = _findings_by_kind(basic_sia.explain())
    assert float(by_kind["partition_margin"][0].value) == pytest.approx(
        float(basic_sia.partition_margin)
    )
    state_margins = by_kind["state_margin"]
    assert len(state_margins) == 2
    assert {f.tone for f in state_margins} == {"cause", "effect"}
    assert by_kind["effectively_tied"][0].value is False


def test_explain_flags_effective_tie(xor_sia):
    by_kind = _findings_by_kind(xor_sia.explain())
    assert by_kind["effectively_tied"][0].value is True


def test_null_sia_explain_has_no_margin_findings():
    by_kind = _findings_by_kind(NullSystemIrreducibilityAnalysis().explain())
    assert "partition_margin" not in by_kind
    assert "state_margin" not in by_kind
    assert "effectively_tied" not in by_kind


def test_to_pandas_includes_margins(basic_sia):
    record = basic_sia.to_pandas()
    assert float(record["partition_margin"]) == pytest.approx(
        float(basic_sia.partition_margin)
    )
    assert float(record["cause_state_margin"]) == pytest.approx(
        float(basic_sia.state_margins[Direction.CAUSE])
    )
    assert float(record["effect_state_margin"]) == pytest.approx(
        float(basic_sia.state_margins[Direction.EFFECT])
    )
    assert bool(record["effectively_tied"]) is False


def test_fig1a_2023_state_margins_match_brute_force():
    """IIT 4.0 (2023) Fig. 1A: the substrate near a specified-state switch
    reports finite, brute-force-consistent state margins.

    The Fig. 1A substrate is not in ``pyphi.examples``; it is built here from
    its published Ising weights. Observed values at the published point
    (default config, IIT_4_0_2023): φ_s = 0.133873, partition_margin =
    0.026941, cause state margin = 0.003492, effect state margin = 0.030059.
    All margins are finite and positive, so the selection is not tied.
    """
    import numpy as np

    from pyphi.substrate_generator import build_substrate
    from pyphi.substrate_generator import ising

    weights = np.array(
        [
            [-0.2, 0.7, 0.2],
            [0.7, -0.2, 0.0],
            [0.0, -0.8, 0.2],
        ]
    )
    substrate = build_substrate([ising.probability] * 3, weights, temperature=0.25)
    sia = pyphi.analyze(substrate, (1, 0, 0), compute="sia")
    assert float(sia.phi) > 0

    system = pyphi.System(substrate, state=(1, 0, 0))
    for direction in Direction.both():
        values = sorted(_per_state_ii(system, direction).values(), reverse=True)
        margin = sia.state_margins[direction]
        assert margin is not None
        assert float(margin) == pytest.approx(values[0] - values[1])
    # Its selections are near a boundary but not tied at the published point.
    assert not sia.effectively_tied
