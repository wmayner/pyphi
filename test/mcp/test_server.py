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
    assert summary["phi"] == pytest.approx(BASIC_PHI_S)
    assert summary["system_phi"] == pytest.approx(BASIC_PHI_S)
    assert summary["big_phi"] == pytest.approx(BASIC_BIG_PHI)
    assert summary["num_distinctions"] == 3
    assert summary["num_relations"] == 4
    assert result["result_ref"].startswith("res")
    assert "Φ" in result["card"]


def test_analyze_sia_only(basic_handle):
    result = srv.analyze(basic_handle, BASIC_STATE, compute="sia")
    assert result["summary"]["phi"] == pytest.approx(BASIC_PHI_S)
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
    assert result["summary"]["phi"] == pytest.approx(0.0)


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
    assert result["summary"]["phi"] == pytest.approx(BASIC_PHI_S)


def test_analyze_guardrail_refuses_large_without_confirmation():
    # An 8-node substrate exceeds the soft threshold for a full analysis; the
    # guard fires before any computation runs.
    tpm = np.zeros((2**8, 8))
    handle = srv.build_substrate(tpm.tolist())["handle"]
    with pytest.raises(ValueError, match="confirm_large"):
        srv.analyze(handle, [0] * 8, compute="full")


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
    }
    for topic in topics:
        assert len(content.load(topic)) > 100


@pytest.mark.skipif(not HAS_EMD, reason="IIT 3.0 needs the emd extra")
def test_analyze_iit3_differs_from_iit4(basic_handle):
    v3 = srv.analyze(basic_handle, BASIC_STATE, formalism="IIT_3_0", compute="sia")
    assert v3["summary"]["phi"] == pytest.approx(0.1875)


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
