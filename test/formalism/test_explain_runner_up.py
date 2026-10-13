"""The runner-up partition and its gap, as ``explain()`` reports them.

The MIP is selected on normalized φ before the intrinsic-information
requirement applies (Mayner et al. 2026, Eqs. 21-23), so the runner-up and
its gap are reported on those same values.
"""

import pytest

import pyphi
from pyphi import examples
from pyphi.models.partitions import concise_partition
from pyphi.partition import system_partitions


def _findings(sia):
    return {f.kind: f.value for f in sia.explain().findings}


@pytest.fixture(scope="module")
def fig1a_ab_sia():
    # Fig 1A, units (A, B): the requirement binds (ii(s) < φ_e), so the
    # reported φ_s is ii(s) while the MIP was chosen on min(φ_c, φ_e).
    with pyphi.config.override(**pyphi.iit4_2026):
        return pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(),
            (0, 1, 1),
            subset=(0, 1),
            compute="sia",
        )


def test_runner_up_gap_is_not_negative_when_the_requirement_binds(fig1a_ab_sia):
    sia = fig1a_ab_sia
    f = _findings(sia)
    assert f["requirement_binding"] == "differentiation"
    assert f["gap"] >= 0
    assert f["runner_up"] != f["winning_partition"]
    # The MIP is unique here, so the gap to the runner-up equals the selection
    # margin, which is also measured before the requirement applies.
    assert f["gap"] == pytest.approx(sia.partition_margin)


def test_requirement_leaves_mip_selection_and_phi_unchanged(fig1a_ab_sia):
    sia = fig1a_ab_sia
    assert float(sia.phi) == pytest.approx(0.04049680811800574)
    assert sia.partition.removed_edges() == frozenset({(0, 1)})
    # The runner-up keeps the values it was ranked on: its normalized φ
    # exceeds the MIP's selection value, min(φ_c, φ_e) over one cut.
    selection_value = min(float(sia.cause.phi), float(sia.effect.phi))
    assert sia.runner_up.normalized_phi == pytest.approx(
        selection_value + sia.runner_up.gap
    )
    assert sia.runner_up.partition.removed_edges() == frozenset({(0, 1), (1, 0)})


def test_runner_up_gap_survives_save_and_load(fig1a_ab_sia, tmp_path):
    path = tmp_path / "sia.json"
    pyphi.save(fig1a_ab_sia, path)
    loaded = pyphi.load(path)
    assert _findings(loaded)["gap"] == pytest.approx(_findings(fig1a_ab_sia)["gap"])


@pytest.mark.parametrize(
    "scheme", ["DIRECTED_SET_PARTITION", "EDGE_CUT_ALL", "EDGE_CUT_BIDIRECTIONAL"]
)
def test_distinct_system_partitions_print_differently(scheme):
    labels = pyphi.labels.NodeLabels(("A", "B", "C"), (0, 1, 2))
    partitions = list(
        system_partitions((0, 1, 2), node_labels=labels, partition_scheme=scheme)
    )
    printed = [concise_partition(p) for p in partitions]
    assert len(set(printed)) == len(partitions)


def test_directed_set_partition_names_severed_connections():
    labels = pyphi.labels.NodeLabels(("A", "B"), (0, 1))
    printed = {
        concise_partition(p)
        for p in system_partitions(
            (0, 1), node_labels=labels, partition_scheme="DIRECTED_SET_PARTITION"
        )
    }
    assert printed == {
        "2 parts: {A,B}; cut A→B",
        "2 parts: {A,B}; cut B→A",
        "2 parts: {A,B}; cut A→B, B→A",
    }


def test_explain_names_the_mip_when_phi_s_is_zero():
    """When the intrinsic-information requirement sets φₛ to 0, explain()
    still names the MIP the runner-up gap is measured from."""
    import numpy as np

    states = pyphi.utils.all_states(3)
    tpm = np.array([[b, c ^ a, a & b] for a, b, c in states], dtype=float)
    substrate = pyphi.Substrate(tpm, node_labels=("A", "B", "C"))
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(substrate, (1, 0, 1), compute="sia")
    f = _findings(sia)
    assert sia.phi == 0
    assert "runner_up" in f
    assert f["winning_partition"] == concise_partition(sia.partition)


def test_explain_names_the_mip_whenever_a_selection_margin_is_reported():
    """A margin or tie flag describes the MIP's selection, so the MIP is named
    even when φₛ is 0 and there is no runner-up."""
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.System(examples.basic_substrate(), (1, 0, 0), subset=(1, 2)).sia()
    f = _findings(sia)
    assert sia.phi == 0
    assert "runner_up" not in f
    assert "partition_margin" in f and "effectively_tied" in f
    assert f["winning_partition"] == concise_partition(sia.partition)
