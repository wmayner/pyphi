"""Tests for selection-margin reporting on the IIT 4.0 SIA."""

import pytest

import pyphi
from pyphi import examples
from pyphi import numerics
from pyphi.conf import config
from pyphi.core import repertoire_algebra as ra
from pyphi.direction import Direction
from pyphi.formalism import queries
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.formalism.iit4 import evaluate_partition
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partitions
from test.conftest import IIT_4_CONFIG


@pytest.fixture(autouse=True)
def _quiet():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield


@pytest.fixture(scope="module")
def basic_sia():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def xor_sia():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
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
    assert numerics.eq(float(xor_sia.state_margins[Direction.CAUSE]), 0.0)
    assert xor_sia.effectively_tied


def test_untied_system_is_not_flagged(basic_sia):
    assert basic_sia.partition_margin is not None
    assert not numerics.eq(float(basic_sia.partition_margin), 0.0)
    assert all(
        margin is None or not numerics.eq(float(margin), 0.0)
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
    from pyphi.conf import presets

    with pyphi.config.override(**presets.iit4_2026):
        sia_2026 = examples.basic_system().sia()
    with pyphi.config.override(**presets.iit4_2023):
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


def _fig1a_substrate(a_to_b=0.7):
    """The IIT 4.0 (2023) Fig. 1A substrate with an adjustable A→B weight."""
    import numpy as np

    from pyphi.substrate_generator import build_substrate
    from pyphi.substrate_generator import ising

    weights = np.array(
        [
            [-0.2, a_to_b, 0.2],
            [0.7, -0.2, 0.0],
            [0.0, -0.8, 0.2],
        ]
    )
    return build_substrate([ising.probability] * 3, weights, temperature=0.25)


def test_partition_margin_none_when_sweep_shortcircuits():
    """At A→B = 0.9 the raw integration is negative, every partition's clamped
    φ is 0, and the sweep stops at the first reducible partition — the margin
    over that truncated prefix is not the true margin, so ``None`` is
    reported."""
    sia = pyphi.analyze(_fig1a_substrate(0.9), (1, 0, 0), compute="sia")
    assert float(sia.phi) == pytest.approx(0.0)
    assert sia.partition_margin is None


def test_partition_margin_exact_without_shortcircuit():
    """With ``shortcircuit_sia=False`` the partition sweep is exhaustive, so
    the margin matches a brute-force sweep even when φ_s = 0. (Here four
    partitions clamp to zero, so the exact margin is an exact tie.)"""
    substrate = _fig1a_substrate(0.9)
    with pyphi.config.override(shortcircuit_sia=False):
        sia = pyphi.analyze(substrate, (1, 0, 0), compute="sia")
        values = _brute_force_partition_values(
            pyphi.System(substrate, state=(1, 0, 0)), sia.system_state
        )
    assert sia.partition_margin is not None
    assert float(sia.partition_margin) == pytest.approx(values[1] - values[0])
    assert float(sia.partition_margin) == pytest.approx(0.0)
    assert sia.effectively_tied


def test_shortcircuit_off_preserves_values(basic_sia):
    """Disabling the short-circuit changes no computed φ value."""
    with pyphi.config.override(shortcircuit_sia=False):
        exhaustive = examples.basic_system().sia()
    assert float(exhaustive.phi) == pytest.approx(float(basic_sia.phi))
    assert float(exhaustive.normalized_phi) == pytest.approx(
        float(basic_sia.normalized_phi)
    )
    assert float(exhaustive.partition_margin) == pytest.approx(
        float(basic_sia.partition_margin)
    )


# ---- mechanism-level margins ----


@pytest.fixture(scope="module")
def basic_mice_effect():
    """The MIE of mechanism (2,) in basic_system — a known purview tie at
    φ = 1 between purviews (1,) and (0, 1)."""
    with pyphi.config.override(progress_bars=False):
        return queries.find_mice(examples.basic_system(), Direction.EFFECT, (2,))


@pytest.fixture(scope="module")
def basic_mice_cause():
    """The MIC of mechanism (0, 2) in basic_system — 10 candidate partitions
    over its purview (0, 1) and a competing purview, so both the partition
    and purview margins are finite and positive."""
    with pyphi.config.override(progress_bars=False):
        return queries.find_mice(examples.basic_system(), Direction.CAUSE, (0, 2))


def _mechanism_partition_values(system, ria):
    """Brute force: normalized φ of every partition of the RIA's
    mechanism-purview pair, at its specified state."""
    return sorted(
        float(
            queries.find_mip(
                system,
                ria.direction,
                ria.mechanism,
                ria.purview,
                partitions=[partition],
                state=ria.specified_state,
            ).normalized_phi
        )
        for partition in mechanism_partitions(
            ria.mechanism, ria.purview, system.node_labels
        )
    )


def test_ria_partition_margin_matches_brute_force(basic_mice_cause):
    ria = basic_mice_cause.ria
    values = _mechanism_partition_values(examples.basic_system(), ria)
    assert len(values) > 1
    assert float(ria.normalized_phi) == pytest.approx(values[0])
    assert ria.partition_margin is not None
    assert float(ria.partition_margin) == pytest.approx(values[1] - values[0])


def test_ria_partition_margin_none_without_competitor(basic_mice_effect):
    # Mechanism (2,) over its winning purview (1,) admits exactly one
    # partition, so there is no competitor and no margin.
    ria = basic_mice_effect.ria
    assert len(list(mechanism_partitions(ria.mechanism, ria.purview, None))) == 1
    assert ria.partition_margin is None


def test_mice_purview_margin_zero_on_purview_tie(basic_mice_effect):
    # Purviews (1,) and (0, 1) tie at φ = 1, so the winner's best competitor
    # matches it exactly.
    assert basic_mice_effect.num_purview_ties >= 1
    assert basic_mice_effect.purview_margin is not None
    assert float(basic_mice_effect.purview_margin) == pytest.approx(0.0)
    assert basic_mice_effect.effectively_tied


def test_mice_purview_margin_matches_brute_force(basic_mice_cause):
    system = examples.basic_system()
    from pyphi.core import repertoire_algebra as ra_kernel

    purviews = ra_kernel.potential_purviews(system, Direction.CAUSE, (0, 2))
    values = sorted(
        (
            float(queries.find_mip(system, Direction.CAUSE, (0, 2), purview).phi)
            for purview in purviews
        ),
        reverse=True,
    )
    assert float(basic_mice_cause.phi) == pytest.approx(values[0])
    assert basic_mice_cause.purview_margin is not None
    assert float(basic_mice_cause.purview_margin) == pytest.approx(values[0] - values[1])


def test_ria_state_margin_reads_specified_state(basic_mice_effect):
    ria = basic_mice_effect.ria
    if ria.specified_state is None or ria.specified_state.state_margin is None:
        pytest.skip("no state competitor for this mechanism")
    assert float(ria.state_margin) == pytest.approx(
        float(ria.specified_state.state_margin)
    )


def test_mechanism_margins_in_pandas_and_findings(basic_mice_effect, basic_mice_cause):
    record = basic_mice_effect.to_pandas()
    assert float(record["purview_margin"]) == pytest.approx(0.0)
    assert "partition_margin" in record
    assert "state_margin" in record
    assert bool(record["effectively_tied"]) is True

    kinds = {f.kind for f in basic_mice_cause.explain().findings}
    assert "purview_margin" in kinds
    assert "partition_margin" in kinds
    assert not basic_mice_cause.effectively_tied


def test_distinction_effectively_tied():
    with pyphi.config.override(progress_bars=False):
        ces = examples.basic_system().ces()
    by_mechanism = {tuple(d.mechanism): d for d in ces.distinctions}
    distinction = by_mechanism[(2,)]
    # Its effect side has the exact purview tie.
    assert distinction.effect.effectively_tied
    assert distinction.effectively_tied


def test_purview_margin_survives_congruence_resolution():
    # A distinction reached through the full cause-effect structure carries the
    # same purview-selection margin as the directly computed MICE. Congruence
    # resolution selects a state-tie peer over the winning purview, which does
    # not itself carry the margin; the margin is a property of the purview
    # choice and must propagate to the selected peer.
    with pyphi.config.override(progress_bars=False):
        system = examples.pqr_system()
        direct = queries.find_mice(system, Direction.CAUSE, (2,))
        ces = system.ces()
    resolved = {tuple(d.mechanism): d for d in ces.distinctions}[(2,)].cause
    assert direct.num_state_ties >= 1
    assert direct.purview_margin is not None
    assert resolved.purview == direct.purview
    assert resolved.purview_margin is not None
    assert float(resolved.purview_margin) == pytest.approx(float(direct.purview_margin))


def test_mechanism_margins_round_trip(basic_mice_cause):
    from pyphi import serialize

    restored = serialize.loads(serialize.dumps(basic_mice_cause))
    assert restored == basic_mice_cause
    assert float(restored.purview_margin) == pytest.approx(
        float(basic_mice_cause.purview_margin)
    )
    assert float(restored.ria.partition_margin) == pytest.approx(
        float(basic_mice_cause.ria.partition_margin)
    )


def test_mechanism_margins_absent_in_old_payloads(basic_mice_cause):
    import json

    from pyphi import serialize

    data = json.loads(serialize.dumps(basic_mice_cause, format="json"))

    def strip(obj):
        if isinstance(obj, dict):
            obj.pop("partition_margin", None)
            obj.pop("purview_margin", None)
            for value in obj.values():
                strip(value)
        elif isinstance(obj, list):
            for item in obj:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.purview_margin is None
    assert restored.ria.partition_margin is None
    assert not restored.effectively_tied


def test_relabel_preserves_mechanism_margins(basic_mice_cause):
    from pyphi import relabel as relabel_mod

    mapping = {0: 2, 1: 0, 2: 1}
    relabeled = relabel_mod.relabel_mice(basic_mice_cause, mapping)
    assert float(relabeled.purview_margin) == pytest.approx(
        float(basic_mice_cause.purview_margin)
    )
    assert float(relabeled.ria.partition_margin) == pytest.approx(
        float(basic_mice_cause.ria.partition_margin)
    )


def test_tied_selections_names_the_tied_selection(xor_sia, basic_sia):
    assert "cause_state" in xor_sia.tied_selections
    assert xor_sia.effectively_tied
    assert basic_sia.tied_selections == ()
    by_kind = _findings_by_kind(xor_sia.explain())
    detail = dict(by_kind["effectively_tied"][0].detail)
    assert "cause_state" in detail["tied_selections"]


def test_tied_selections_partition(basic_sia):
    sia = examples.grid3_system().sia()
    assert "partition" in sia.tied_selections
    assert basic_sia.tied_selections == ()
