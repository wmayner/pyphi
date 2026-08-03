"""Tests for the PyPhi MCP server tools.

Drive the tools in-process and assert against known Φ values from the IIT 4.0
paper's basic example, plus the structural behavior of the registry, guardrail,
and reference material.
"""

import importlib.util

import numpy as np
import pytest

import pyphi

pytest.importorskip("mcp")

from pyphi.mcp import content
from pyphi.mcp import server as srv
from test.conftest import IIT_4_CONFIG

HAS_EMD = importlib.util.find_spec("ot") is not None
HAS_VIZ = importlib.util.find_spec("matplotlib") is not None and (
    importlib.util.find_spec("plotly") is not None
)

# Known values for basic_substrate in state (1, 1, 0) under IIT 4.0 (2023).
# Phi reflects the S1 state-tie resolution: among specified-state readings
# tied at phi_s, Composition selects the Phi-maximal reading.
BASIC_STATE = [1, 1, 0]
BASIC_PHI_S = 0.20751874963942188
BASIC_BIG_PHI = 2.0501249975961455


@pytest.fixture(autouse=True)
def _quiet():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield


@pytest.fixture
def basic_handle():
    return srv.load_example("basic")["handle"]


def test_list_examples_includes_standard_networks():
    examples = srv.list_examples()
    assert "basic" in examples
    assert "xor" in examples
    assert all(isinstance(desc, str) for desc in examples.values())


def test_load_and_describe_substrate(basic_handle):
    info = srv.describe_substrate(basic_handle)
    assert info["num_nodes"] == 3
    assert info["node_labels"] == ["A", "B", "C"]
    assert np.array(info["connectivity_matrix"]).shape == (3, 3)
    assert "little-endian" in info["state_convention"]


def test_analyze_basic_known_phi(basic_handle):
    result = srv.analyze(basic_handle, BASIC_STATE)
    summary = result["summary"]
    assert summary["system_phi"] == pytest.approx(BASIC_PHI_S)
    # φₛ and Φ are different quantities; the summary must not conflate them.
    assert "phi" not in summary
    assert summary["system_phi"] != summary["big_phi"]
    assert summary["big_phi"] == pytest.approx(BASIC_BIG_PHI)
    assert summary["num_distinctions"] == 3
    assert summary["num_relations"] == 4
    assert result["result_ref"].startswith("res")
    assert "Φ" in result["card"]


def test_analyze_sia_only(basic_handle):
    result = srv.analyze(basic_handle, BASIC_STATE, compute="sia")
    assert result["summary"]["system_phi"] == pytest.approx(BASIC_PHI_S)
    # A system irreducibility analysis has no cause-effect structure counts.
    assert "num_distinctions" not in result["summary"]


def test_analyze_ces_only(basic_handle):
    result = srv.analyze(basic_handle, BASIC_STATE, compute="ces")
    assert result["summary"]["big_phi"] == pytest.approx(BASIC_BIG_PHI)
    assert result["summary"]["num_distinctions"] == 3


def test_analyze_2026_differentiation_cap(basic_handle):
    # A deterministic system provides no repertoire of alternatives, so the
    # 2026 differentiation requirement drives its φₛ to zero.
    result = srv.analyze(
        basic_handle, BASIC_STATE, formalism="IIT_4_0_2026", compute="sia"
    )
    assert result["summary"]["system_phi"] == pytest.approx(0.0)


def test_inspect_distinction(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE)["result_ref"]
    detail = srv.inspect(ref, "ces.distinctions[0]")
    assert detail["type"] == "Distinction"
    assert detail["serialized"] is not None


def test_inspect_serializes_sia(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE)["result_ref"]
    detail = srv.inspect(ref, "sia")
    assert detail["type"] == "SystemIrreducibilityAnalysis"
    assert detail["serialized"] is not None


def test_detail_full_embeds_serialization(basic_handle):
    result = srv.analyze(basic_handle, BASIC_STATE, detail="full")
    assert result["serialized"] is not None
    assert len(result["serialized"]) > 0


def test_build_substrate_from_tpm():
    # The basic 3-node system's state-by-node TPM (A=OR, B=COPY, C=XOR of the
    # others), rows in little-endian state order.
    tpm = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ]
    built = srv.build_substrate(tpm, node_labels=["A", "B", "C"])
    assert built["num_nodes"] == 3
    result = srv.analyze(built["handle"], BASIC_STATE)
    assert result["summary"]["system_phi"] == pytest.approx(BASIC_PHI_S)


def test_analyze_guardrail_refuses_large_without_confirmation():
    # An 8-node substrate exceeds the soft threshold for a full analysis; the
    # guard fires before any computation runs.
    tpm = np.zeros((2**8, 8))
    handle = srv.build_substrate(tpm.tolist())["handle"]
    with pytest.raises(ValueError, match="confirm_large"):
        srv.analyze(handle, [0] * 8, compute="full")


def test_analyze_guard_reports_estimated_counts():
    tpm = np.zeros((2**8, 8))
    handle = srv.build_substrate(tpm.tolist())["handle"]
    with pytest.raises(ValueError, match="mechanism-partition sweeps"):
        srv.analyze(handle, [0] * 8, compute="full")


def test_count_gate_admits_sparse_system_above_old_node_limit():
    # Eight disconnected units have no candidate purviews at all, so the
    # estimated workload is trivial and the guard admits the analysis
    # without confirmation, where a node-count guard refused at this size.
    tpm = np.zeros((2**8, 8))
    cm = np.zeros((8, 8))
    handle = srv.build_substrate(tpm.tolist(), cm=cm.tolist())["handle"]
    out = srv.analyze(handle, [0] * 8, compute="ces")
    assert "result_ref" in out


def test_estimate_cost_tool(basic_handle):
    out = srv.estimate_cost(basic_handle)
    assert "AnalysisEstimate" in out["card"]
    est = out["estimate"]
    assert est["n_units"] == 3
    assert est["mechanisms"] == 7
    assert est["capped"] is False


def test_estimate_cost_sia_scope(basic_handle):
    est = srv.estimate_cost(basic_handle, compute="sia")["estimate"]
    assert est["system_partitions"] == 22
    assert est["mechanism_partition_sweeps"] is None


def test_estimate_cost_unknown_formalism_is_a_clear_error(basic_handle):
    with pytest.raises(ValueError, match="unknown formalism"):
        srv.estimate_cost(basic_handle, formalism="IIT_5_0")


def test_unknown_handle_is_a_clear_error():
    with pytest.raises(KeyError, match="Unknown substrate handle"):
        srv.describe_substrate("nope")


def test_reference_topics_load():
    topics = content.topics()
    assert set(topics) == {
        "primer",
        "theory",
        "equations",
        "gotchas",
        "interpreting-results",
        "building-systems",
        "migration",
        "configuration",
        "performance",
        "parallelization",
        "campaigns",
        "visualization",
    }
    for topic in topics:
        assert len(content.load(topic)) > 100


def test_migration_topic_covers_the_renames():
    doc = content.load("migration")
    # The load-bearing facts an agent needs, guarding against a stub file.
    assert "no deprecation shims" in doc
    for name in ("Substrate", "System", "analyze", "IIT_3_0"):
        assert name in doc


def test_visualization_topic_covers_the_gotchas():
    doc = content.load("visualization")
    # The facts a fresh agent gets wrong without them, guarding against a stub.
    for name in ("plot_ces", "max_relations", "ANALYTICAL", "hypergraph", "write_html"):
        assert name in doc


def test_parallelization_topic_covers_the_gates():
    doc = content.load("parallelization")
    # The load-bearing facts: the two-gate rule, the per-level options, the
    # measured guidance, and the tool surface, guarding against a stub.
    for name in (
        "sequential_threshold",
        "parallel_distinction_evaluation",
        "parallel_partition_evaluation",
        "override",
        "loky",
        "configure_parallel",
    ):
        assert name in doc
    # The global flag alone engages nothing — the most common mistake.
    assert "not sufficient" in doc


def test_analyze_parallel_matches_sequential(basic_handle):
    # Parallelism never changes a result. (This small system falls below
    # every sequential_threshold, so the test exercises the override
    # plumbing and result invariance, not worker-pool spin-up.)
    result = srv.analyze(basic_handle, BASIC_STATE, parallel=True, workers=2)
    assert result["summary"]["system_phi"] == pytest.approx(BASIC_PHI_S)
    assert result["summary"]["big_phi"] == pytest.approx(BASIC_BIG_PHI)
    # The per-call override is scoped: the configuration is restored.
    assert pyphi.config.infrastructure.parallel is False


def test_analyze_parallel_unknown_level_errors(basic_handle):
    with pytest.raises(ValueError, match="Unknown parallel level"):
        srv.analyze(basic_handle, BASIC_STATE, parallel=["bogus"])


def test_configure_parallel_roundtrip():
    # A no-argument call reads without changing anything.
    state = srv.configure_parallel()
    assert state["parallel"] is False
    assert state["workers"] == -1

    state = srv.configure_parallel(enable=True, levels=["partitions"], workers=2)
    assert state["parallel"] is True
    assert state["workers"] == 2
    assert state["levels"]["partitions"]["parallel"] is True
    assert state["levels"]["relations"]["parallel"] is False
    # A subsequent read agrees with what was set.
    assert srv.configure_parallel() == state

    state = srv.configure_parallel(reset=True)
    assert state["parallel"] is False
    assert state["workers"] == -1
    assert all(not level["parallel"] for level in state["levels"].values())


def test_analyze_parallel_false_overrides_server_config(basic_handle):
    srv.configure_parallel(enable=True)
    result = srv.analyze(basic_handle, BASIC_STATE, compute="sia", parallel=False)
    assert result["summary"]["system_phi"] == pytest.approx(BASIC_PHI_S)
    # The per-call setting is scoped; the server configuration persists.
    assert pyphi.config.infrastructure.parallel is True


def test_analyze_guardrail_unchanged_with_parallel():
    # Parallelism divides constants, not exponents: the size guard still fires.
    tpm = np.zeros((2**8, 8))
    handle = srv.build_substrate(tpm.tolist())["handle"]
    with pytest.raises(ValueError, match="confirm_large"):
        srv.analyze(handle, [0] * 8, compute="full", parallel=True)


@pytest.mark.skipif(not HAS_EMD, reason="IIT 3.0 needs the emd extra")
def test_analyze_iit3_differs_from_iit4(basic_handle):
    v3 = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_3_0", compute="sia")
    assert v3["summary"]["system_phi"] == pytest.approx(0.1875)


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_ces_is_html_only(basic_handle):
    # The interactive Φ-structure returns an HTML path and never an inline
    # image: a static snapshot would be misleading and would suppress discovery
    # of the interactive file.
    ref = srv.analyze(basic_handle, BASIC_STATE, compute="ces")["result_ref"]
    out = srv.plot(ref, kind="ces")
    assert isinstance(out, str)
    assert ".html" in out
    assert "interactive" in out.lower()


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
@pytest.mark.parametrize("kind", ["connectivity", "tpm"])
def test_plot_substrate_figures(basic_handle, kind):
    out = srv.plot(basic_handle, kind=kind)
    assert isinstance(out, list)  # a message plus an inline PNG
    assert ".png" in out[0]


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_repertoires(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE)["result_ref"]
    out = srv.plot(ref, kind="repertoires")
    assert isinstance(out, list)
    assert ".png" in out[0]


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_unknown_kind_errors(basic_handle):
    with pytest.raises(ValueError, match="Unknown plot kind"):
        srv.plot(basic_handle, kind="bogus")


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_ces_view(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE, compute="ces")["result_ref"]
    out = srv.plot(ref, kind="ces", view="hypergraph")
    assert isinstance(out, str)
    assert ".html" in out


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_ces_unknown_view_errors(basic_handle):
    # "barycentric" is a layout, not a view — the common confusion.
    ref = srv.analyze(basic_handle, BASIC_STATE, compute="ces")["result_ref"]
    with pytest.raises(ValueError, match="Unknown view"):
        srv.plot(ref, kind="ces", view="barycentric")


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_ces_analytical_defaults_to_max_relations(basic_handle):
    with pyphi.config.override(relation_computation="ANALYTICAL"):
        ref = srv.analyze(basic_handle, BASIC_STATE, compute="ces")["result_ref"]
        # Analytical relations now default to rendering the strongest 1000.
        out_default = srv.plot(ref, kind="ces")
        assert isinstance(out_default, str)
        assert ".html" in out_default
        # Explicit cap still works.
        out_capped = srv.plot(ref, kind="ces", max_relations=8)
        assert isinstance(out_capped, str)
        assert ".html" in out_capped


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_view_rejected_for_non_ces(basic_handle):
    with pytest.raises(ValueError, match="only to kind='ces'"):
        srv.plot(basic_handle, kind="tpm", view="hypergraph")


def _kary_joint_tpm_lists():
    """An explicit-alphabet joint TPM for a two-unit ternary substrate,
    as nested lists (the MCP tool's wire format)."""
    rng = np.random.default_rng(7)

    def marginal():
        m = rng.random((3, 3, 3))
        return m / m.sum(axis=-1, keepdims=True)

    substrate = pyphi.Substrate(
        marginals=[marginal(), marginal()], state_space=(0, 1, 2)
    )
    return np.asarray(substrate.tpm.to_joint()).tolist()


def test_build_substrate_kary_alphabet_list():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    assert out["num_nodes"] == 2
    assert srv._get_substrate(out["handle"]).num_states == 9


def test_state_by_state_binary_output_unchanged(basic_handle):
    from pyphi import convert

    substrate = srv._get_substrate(basic_handle)
    on = np.asarray(substrate.tpm.to_joint())[..., 1]
    expected = convert.state_by_node2state_by_state(
        on.reshape(-1, substrate.size, order="F")
    )
    assert np.allclose(srv._state_by_state(substrate), expected)


def test_state_by_state_kary():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    substrate = srv._get_substrate(out["handle"])
    sbs = srv._state_by_state(substrate)
    assert sbs.shape == (9, 9)
    assert np.allclose(sbs.sum(axis=1), 1.0)


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_tpm_kary_substrate():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    result = srv.plot(out["handle"], kind="tpm")
    assert isinstance(result, list)
    assert ".png" in result[0]


class TestCampaignTools:
    def test_prepare_status_collect_roundtrip(self, tmp_path):
        handle = srv.load_example("basic")["handle"]
        directory = tmp_path / "camp"
        prepared = srv.prepare_campaign(
            handles=[handle],
            states="all",
            formalisms=["IIT_4_0_2026"],
            directory=str(directory),
            jobs=2,
        )
        assert prepared["status"]["n_tasks"] == 2
        assert "card" in prepared

        # Execute the tasks locally (the runner, not condor).
        from pyphi.campaign.runner import run_task

        for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
            assert (
                run_task(
                    task_file,
                    substrates_dir=directory / "substrates",
                    outputs_dir=directory / "outputs",
                )
                == 0
            )

        st = srv.campaign_status(directory=str(directory))
        # dataclasses.asdict preserves tuples in-process; check emptiness,
        # not list equality.
        assert not st["status"]["failed"]
        assert not st["status"]["pending"]

        collected = srv.collect_campaign(directory=str(directory))
        assert "result_ref" in collected
        assert collected["rows"] >= 1


class TestCESCampaignTools:
    def test_ces_campaign_roundtrip(self, tmp_path):
        handle = srv.load_example("basic")["handle"]
        directory = tmp_path / "ces-camp"
        prepared = srv.prepare_ces_campaign(
            handle=handle,
            state=[1, 0, 0],
            formalism="IIT_4_0_2026",
            directory=str(directory),
            units_per_job=50.0,
            scope={"mechanisms": {"max_order": 2}},
        )
        assert prepared["status"]["n_tasks"] >= 1

        from pyphi.campaign.runner import run_task

        for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
            assert (
                run_task(
                    task_file,
                    substrates_dir=directory / "substrates",
                    outputs_dir=directory / "outputs",
                )
                == 0
            )
        collected = srv.collect_campaign(directory=str(directory))
        assert collected["type"] == "CauseEffectStructure"
        assert "scope_report" in collected

    def test_estimate_cost_scope(self):
        handle = srv.load_example("basic")["handle"]
        full = srv.estimate_cost(handle, compute="ces")["estimate"]
        scoped = srv.estimate_cost(
            handle, compute="ces", scope={"mechanisms": {"max_order": 1}}
        )["estimate"]
        assert scoped["mechanisms"] < full["mechanisms"]
