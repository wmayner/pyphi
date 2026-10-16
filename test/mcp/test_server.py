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


def test_list_examples_descriptions_are_whole_sentences():
    examples = srv.list_examples()
    assert examples["actual_causation"] == (
        "The actual causation example substrate, consisting of an OR and AND "
        "gate with self-loops."
    )
    assert all("``" not in text and "\n" not in text for text in examples.values())


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
        "documentation",
        "reproducible-work",
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


def _call_tool(name, arguments):
    import anyio

    return anyio.run(srv.mcp.call_tool, name, arguments)


@pytest.mark.parametrize(
    ("tool", "arguments", "unknown"),
    [
        ("analyze", {"state": [1, 0, 0], "formalsim": "iit3"}, "formalsim"),
        ("analyze", {"state": [1, 0, 0], "subsets": [0, 1]}, "subsets"),
        ("describe_substrate", {"bogus": 1}, "bogus"),
    ],
)
def test_tools_reject_unknown_arguments(basic_handle, tool, arguments, unknown):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match=unknown):
        _call_tool(tool, {"handle": basic_handle, **arguments})


def test_prepare_campaign_rejects_a_seed(basic_handle, tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    arguments = {
        "handles": [basic_handle],
        "states": [[1, 0, 0]],
        "directory": str(tmp_path / "campaign"),
        "jobs": 1,
        "seed": 3,
    }
    with pytest.raises(ToolError, match="seed"):
        _call_tool("prepare_campaign", arguments)


def test_tool_schemas_forbid_unknown_arguments():
    import anyio

    tools = anyio.run(srv.mcp.list_tools)
    assert tools
    assert all(tool.inputSchema.get("additionalProperties") is False for tool in tools)


def test_a_known_argument_still_reaches_the_tool(basic_handle):
    _, structured = _call_tool(
        "analyze", {"handle": basic_handle, "state": [1, 0, 0], "subset": [0, 1]}
    )
    assert structured["summary"]["subset"] == [0, 1]


@pytest.mark.parametrize("tool", ["analyze", "estimate_cost"])
def test_a_boolean_mask_subset_is_rejected(basic_handle, tool):
    # JSON true/false must not be read as the indices 1 and 0.
    from mcp.server.fastmcp.exceptions import ToolError

    arguments = {"handle": basic_handle, "subset": [False, True]}
    if tool == "analyze":
        arguments["state"] = [1, 0, 0]
    with pytest.raises(ToolError, match="subset"):
        _call_tool(tool, arguments)


@pytest.mark.parametrize("tool", ["analyze", "estimate_cost"])
def test_a_subset_of_integral_floats_is_accepted(basic_handle, tool):
    # Some JSON clients send every number as a float.
    from mcp.server.fastmcp.exceptions import ToolError

    arguments = {"handle": basic_handle, "subset": [0.0, 1.0]}
    if tool == "analyze":
        arguments["state"] = [1, 0, 0]
    _, structured = _call_tool(tool, arguments)
    if tool == "analyze":
        assert structured["summary"]["subset"] == [0, 1]
    with pytest.raises(ToolError, match="subset"):
        _call_tool(tool, {**arguments, "subset": [0.5, 1.0]})


def test_configure_parallel_rejects_zero_workers_and_changes_nothing():
    before = srv.configure_parallel()
    with pytest.raises(ValueError, match="parallel_workers"):
        srv.configure_parallel(enable=True, workers=0)
    assert srv.configure_parallel() == before


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


def test_instructions_carry_the_gotchas():
    """The gotchas load with the primer rather than waiting behind a tool call.

    An instruction to go read them is skipped exactly when the task feels too
    small to need theory, which is when these mistakes get made.
    """
    from pyphi.mcp.server import _INSTRUCTIONS

    assert "φₛ and Φ are different quantities" in _INSTRUCTIONS
    assert "States are little-endian" in _INSTRUCTIONS
    assert "no possible predecessor" in _INSTRUCTIONS
    # The primer's own abbreviated version is gone, not duplicated.
    assert "Two things to keep straight" not in _INSTRUCTIONS


def test_analyze_distinctions_only(basic_handle):
    # (1, 0, 0) leaves the system's specified state untied, so the congruent
    # distinctions are exactly the cause-effect structure's.
    out = srv.analyze(basic_handle, [1, 0, 0], compute="distinctions")
    ces = srv.analyze(basic_handle, [1, 0, 0], compute="ces")["summary"]
    summary = out["summary"]
    assert summary["congruence"] == "resolved"
    assert summary["num_distinctions"] == ces["num_distinctions"]
    assert summary["sum_phi_distinctions"] == pytest.approx(ces["sum_phi_distinctions"])
    # No Φ is claimed: relations were never computed.
    assert "big_phi" not in summary
    assert "num_relations" not in summary


def test_analyze_distinctions_flags_an_unresolved_count_as_an_upper_bound(
    basic_handle,
):
    # (1, 1, 0) ties the specified state, so congruence stays undetermined.
    # The counts must not be reportable as the structure's: congruence
    # filtering can drop any number of distinctions, including all of them.
    summary = srv.analyze(basic_handle, [1, 1, 0], compute="distinctions")["summary"]
    assert summary["congruence"] == "unresolved"
    assert "num_distinctions" not in summary
    assert "sum_phi_distinctions" not in summary
    assert summary["num_distinctions_upper_bound"] >= 0
    assert "upper bounds" in summary["note"]


def test_analyze_distinctions_under_iit3_reports_the_structure(basic_handle):
    # IIT 3.0 has no congruence filter, so the key would name a step that
    # formalism never takes.
    summary = srv.analyze(
        basic_handle, BASIC_STATE, formalism="IIT_3_0", compute="distinctions"
    )["summary"]
    assert "congruence" not in summary
    assert summary["num_distinctions"] > 0


def test_iit3_full_analysis_reports_its_distinction_counts(basic_handle):
    summary = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_3_0")["summary"]
    assert summary["num_distinctions"] > 0
    assert summary["sum_phi_distinctions"] > 0
    # IIT 3.0 has no relations, so no structure integrated information.
    assert "big_phi" not in summary
    assert "num_relations" not in summary
    assert "sum_phi_relations" not in summary


def test_guard_charges_a_ces_analysis_the_system_partition_axis(
    basic_handle, monkeypatch
):
    # Under IIT 4.0 a cause-effect structure computes a system irreducibility
    # analysis first, so the guard must weigh that axis for 'ces' too — a
    # sparse substrate can be trivial on the distinction axis and enormous on
    # this one.
    monkeypatch.setattr(srv, "_SIA_PARTITION_LIMIT", 1)
    with pytest.raises(ValueError, match="system partitions"):
        srv.analyze(basic_handle, BASIC_STATE, compute="ces")
    # The refusal points at the analysis that skips that axis.
    with pytest.raises(ValueError, match="compute='distinctions'"):
        srv.analyze(basic_handle, BASIC_STATE, compute="ces")
    # ...which is itself admitted, since it never sweeps a system partition.
    assert "result_ref" in srv.analyze(basic_handle, BASIC_STATE, compute="distinctions")


def test_guard_charges_an_iit3_ces_analysis_the_system_partition_axis(
    basic_handle, monkeypatch
):
    # The IIT 3.0 cause-effect structure embeds its system irreducibility
    # analysis too.
    monkeypatch.setattr(srv, "_SIA_PARTITION_LIMIT", 1)
    with pytest.raises(ValueError, match="system partitions"):
        srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_3_0", compute="ces")


def test_estimate_cost_distinctions_scope(basic_handle):
    est = srv.estimate_cost(basic_handle, compute="distinctions")["estimate"]
    assert est["system_partitions"] is None
    assert est["mechanism_partition_sweeps"] > 0


def _estimate(**overrides):
    """An AnalysisEstimate with every axis trivial except those overridden."""
    from pyphi.cost import AnalysisEstimate

    fields = {
        "n_units": 16,
        "state_space_size": 2**16,
        "compute": "ces",
        "system_partitions": 1,
        "specified_state_evaluations": 1,
        "mechanisms": 1,
        "purview_evaluations": 1,
        "mechanism_partition_sweeps": 1,
        "relations_closed_form": True,
        "possible_distinctions": 1,
        "possible_relations": 1,
        "capped": False,
    }
    return AnalysisEstimate(**{**fields, **overrides})


def test_guard_checks_the_specified_state_axis_on_its_own():
    """The axis that dominates a large system's cost is checked in isolation.

    Over sparse connectivity every mechanism-oriented axis can stay modest
    while the specified-state search still sweeps the whole state space, so an
    axis-blind guard admits a run that spends its time in a single call.
    """
    estimate = _estimate(specified_state_evaluations=2 * 2**17)
    with pytest.raises(ValueError, match="262,144 specified-state evaluations"):
        srv._refuse_if_large(estimate, "ces", 17)


def test_guard_offers_the_distinction_sweep_when_the_search_is_the_problem():
    """A bare distinction sweep performs no specified-state search."""
    estimate = _estimate(specified_state_evaluations=2 * 2**17)
    with pytest.raises(ValueError, match="compute='distinctions'"):
        srv._refuse_if_large(estimate, "ces", 17)


def test_guard_admits_a_specified_state_search_within_the_limit():
    # Sixteen units: 131,072 evaluations, at the limit — about a minute.
    srv._refuse_if_large(_estimate(specified_state_evaluations=2 * 2**16), "ces", 16)


def test_the_specified_state_limit_matches_the_kernels_own_sweep_bound():
    """The guard refuses exactly what the sweeps would refuse, but earlier.

    A looser limit here would admit a call the kernel then rejects partway
    through its first sweep; a tighter one would refuse sizes that complete.
    """
    from pyphi.core.repertoire_algebra import _MAX_FORWARD_SWEEP_STATES

    assert srv._SPECIFIED_STATE_LIMIT == 2 * _MAX_FORWARD_SWEEP_STATES


def test_guard_ignores_the_axis_when_the_formalism_has_no_specified_state():
    srv._refuse_if_large(_estimate(specified_state_evaluations=None), "ces", 16)


def test_analyze_subset_analyzes_the_candidate_system():
    """The Fig 1A pair aB reproduces the paper's 0.17 under the 2023 formalism
    through the tool."""
    handle = srv.load_example("iit4_2023_fig1a")["handle"]
    out = srv.analyze(
        handle, [0, 1, 1], subset=["A", "B"], formalism="IIT_4_0_2023", compute="sia"
    )
    assert round(out["summary"]["system_phi"], 2) == 0.17
    assert out["summary"]["subset"] == ["A", "B"]


def test_analyze_summary_carries_mip_and_requirement_terms(basic_handle):
    out = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_4_0_2026")
    summary = out["summary"]
    assert summary["mip"]
    assert summary["intrinsic_information"] == 0.0
    assert summary["requirement_binding"] == {
        "term": "differentiation",
        "direction": "EFFECT",
    }
    assert "MIP" in out["card"] and "ii(s)" in out["card"]
    # The 2023 version does not use ii(s) in φₛ, so its summary, like its
    # card, carries neither ii(s) nor a binding term.
    out_2023 = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_4_0_2023")
    assert "intrinsic_information" not in out_2023["summary"]
    assert "requirement_binding" not in out_2023["summary"]
    assert "ii(s)" not in out_2023["card"]


def test_analyze_says_why_a_system_that_is_not_strongly_connected_has_zero_phi(
    basic_handle,
):
    out = srv.analyze(basic_handle, BASIC_STATE, subset=["A", "B"], compute="sia")
    assert out["summary"]["null_reason"] == "not strongly connected"
    assert "not strongly connected" in out["card"]
    assert "None" not in out["card"]


def test_estimate_cost_states_what_the_seconds_cover(basic_handle):
    full = srv.estimate_cost(basic_handle, compute="full")
    sia = srv.estimate_cost(basic_handle, compute="sia")
    assert "system-partition" in full["estimated_cpu_seconds_covers"]
    assert sia["estimated_cpu_seconds"] is None
    assert "system-partition" in sia["estimated_cpu_seconds_covers"]


def test_substrate_summary_carries_the_tpm_as_rows(basic_handle):
    tpm = srv.describe_substrate(basic_handle)["tpm"]
    assert tpm["form"] == "state-by-node"
    assert tpm["units"] == ["A", "B", "C"]
    rows = {tuple(r["state"]): r["p_on"] for r in tpm["rows"]}
    assert len(rows) == 8
    # A is the OR of B and C, B copies C, C is the XOR of A and B.
    for (a, b, c), p_on in rows.items():
        assert p_on == [float(b or c), float(c), float(a ^ b)]


def test_multivalued_summary_uses_per_unit_distributions():
    handle = srv.load_example("gomez_p53_mdm2")["handle"]
    tpm = srv.describe_substrate(handle)["tpm"]
    assert tpm["alphabet_sizes"] == [3, 2, 2]
    row = tpm["rows"][0]
    assert "p_next" in row
    assert [len(d) for d in row["p_next"]] == [3, 2, 2]
    assert all(abs(sum(d) - 1.0) < 1e-9 for d in row["p_next"])


def test_large_substrate_summary_omits_the_tpm():
    n = 9  # 512 states, above the limit
    handle = srv.build_substrate(tpm=[[0] * n for _ in range(2**n)])["handle"]
    tpm = srv.describe_substrate(handle)["tpm"]
    assert "rows" not in tpm
    assert "512 states" in tpm["omitted"]


def test_describe_substrate_reports_the_pyphi_version(basic_handle):
    from importlib.metadata import version

    assert srv.describe_substrate(basic_handle)["pyphi_version"] == version("pyphi")


def test_guard_refuses_a_fully_connected_10_unit_substrate_quickly():
    """The 9-unit partition count already exceeds the guard's counting budget,
    so the 10-unit count is refused without enumerating any partition."""
    import time

    handle = srv.build_substrate(np.zeros((2**10, 10)).tolist())["handle"]
    start = time.perf_counter()
    for compute in ("sia", "full"):
        with pytest.raises(ValueError, match="at least 4,419,572 system partitions"):
            srv.analyze(handle, [0] * 10, compute=compute)
    assert time.perf_counter() - start < 10


def test_estimate_cost_prices_the_subset(basic_handle):
    whole = srv.estimate_cost(basic_handle)["estimate"]
    pair = srv.estimate_cost(basic_handle, subset=["A", "B"])["estimate"]
    assert (whole["n_units"], pair["n_units"]) == (3, 2)
    assert pair["mechanisms"] == 3


@pytest.mark.parametrize(
    ("scope", "named"),
    [
        ({"mechanisms": {"max_ordr": 1}}, "max_ordr"),
        ({"mechanism": {"max_order": 1}}, "mechanism"),
    ],
)
def test_an_unknown_scope_key_is_rejected(basic_handle, scope, named):
    with pytest.raises(ValueError, match=named):
        srv.estimate_cost(basic_handle, compute="distinctions", scope=scope)


def test_scope_bounds_purview_order_by_mechanism_order(basic_handle):
    full = srv.estimate_cost(basic_handle, compute="distinctions")["estimate"]
    table = [[1, 1], [2, 1], [3, 1]]
    capped = srv.estimate_cost(
        basic_handle,
        compute="distinctions",
        scope={"max_purview_order_by_mechanism_order": table},
    )["estimate"]
    assert capped["purview_evaluations"] < full["purview_evaluations"]


def test_inspect_without_a_path_returns_the_summary_and_the_paths(basic_handle):
    import json

    ref = srv.analyze(basic_handle, BASIC_STATE)["result_ref"]
    out = srv.inspect(ref)
    assert len(json.dumps(out)) < 5000
    assert "serialized" not in out
    assert (
        out["summary"]["system_phi"]
        == srv.analyze(basic_handle, BASIC_STATE)["summary"]["system_phi"]
    )
    assert {"sia", "ces", "ces.relations"} <= set(out["paths"])
    for path in out["paths"]:
        srv.inspect(ref, path)
    for path, count in out["indexed"].items():
        assert count > 0
        srv.inspect(ref, f"{path}[{count - 1}]")


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_repertoires_of_a_sia_result_names_compute(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE, compute="sia")["result_ref"]
    with pytest.raises(ValueError, match="compute"):
        srv.plot(ref, kind="repertoires")


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_ces_of_bare_distinctions_names_compute(basic_handle):
    ref = srv.analyze(basic_handle, [1, 0, 0], compute="distinctions")["result_ref"]
    with pytest.raises(ValueError, match="compute"):
        srv.plot(ref, kind="ces")


@pytest.mark.skipif(not (HAS_VIZ and HAS_EMD), reason="needs visualize and emd")
def test_plot_ces_draws_an_iit3_structure(basic_handle):
    ref = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_3_0")["result_ref"]
    assert ".html" in srv.plot(ref, kind="ces")


@pytest.mark.skipif(not HAS_EMD, reason="IIT 3.0 needs the emd extra")
def test_confirm_large_still_canonicalizes_the_formalism(basic_handle):
    out = srv.analyze(
        basic_handle,
        [1, 0, 0],
        formalism="iit3",
        compute="distinctions",
        confirm_large=True,
    )
    assert out["summary"]["formalism"] == "IIT_3_0"
    assert "congruence" not in out["summary"]


def test_an_unknown_detail_is_rejected(basic_handle):
    with pytest.raises(ValueError, match="detail"):
        srv.analyze(basic_handle, BASIC_STATE, detail="bogus")


def test_an_unknown_compute_names_the_tools_vocabulary(basic_handle):
    with pytest.raises(ValueError, match="'full'"):
        srv.analyze(basic_handle, BASIC_STATE, compute="bogus")


def test_default_labels_and_state_convention_follow_the_substrate():
    out = srv.build_substrate(tpm=[[0, 0], [1, 0], [0, 1], [1, 1]])
    assert out["node_labels"] == ["n0", "n1"]
    convention = srv.describe_substrate(out["handle"])["state_convention"]
    assert "(1, 0) means n0 on, n1 off" in convention


def test_a_kary_alphabet_with_a_two_dimensional_tpm_names_the_joint_form():
    with pytest.raises(ValueError, match="n_units, max_alphabet") as error:
        srv.build_substrate(tpm=np.full((6, 2), 0.5).tolist(), alphabet=[3, 2])
    assert "marginals" not in str(error.value)


def test_campaign_tools_take_no_seed():
    import inspect

    for tool in (srv.prepare_campaign, srv.prepare_ces_campaign):
        assert "seed" not in inspect.signature(tool).parameters, tool


def test_prepare_campaign_accepts_the_full_analysis(tmp_path):
    handle = srv.load_example("basic")["handle"]
    prepared = srv.prepare_campaign(
        handles=[handle],
        states=[[1, 1, 0]],
        directory=str(tmp_path / "camp"),
        compute="full",
    )
    assert prepared["status"]["n_tasks"] == 1


def test_prepare_ces_campaign_without_a_formalism(tmp_path):
    handle = srv.load_example("basic")["handle"]
    prepared = srv.prepare_ces_campaign(
        handle=handle,
        state=[1, 0, 0],
        directory=str(tmp_path / "camp"),
        units_per_job=1e6,
    )
    assert prepared["status"]["n_tasks"] >= 1


def test_prepare_campaign_reads_a_flat_list_as_one_state_or_subset(tmp_path):
    handle = srv.load_example("basic")["handle"]
    prepared = srv.prepare_campaign(
        handles=[handle],
        states=[1, 0, 0],
        subsets=[0, 1],
        directory=str(tmp_path / "camp"),
    )
    assert prepared["status"]["n_cells"] == 1


def test_guard_admits_an_8_unit_system_analysis_and_refuses_9():
    """About 1 ms per system partition: 8 fully connected units (510,313
    partitions) take minutes, 9 (4,419,572) take over an hour."""
    import pyphi

    for n, refused in ((8, False), (9, True)):
        substrate = pyphi.Substrate(np.full((2**n, n), 0.5), cm=np.ones((n, n)))
        with pyphi.config.override(**pyphi.iit4_2026):
            estimate = pyphi.estimate_analysis(substrate, (0,) * n, compute="sia")
        if refused:
            with pytest.raises(ValueError, match="system partitions"):
                srv._refuse_if_large(estimate, "sia", n)
        else:
            srv._refuse_if_large(estimate, "sia", n)
