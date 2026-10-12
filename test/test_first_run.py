"""What someone new to PyPhi 2.0 meets first: names carried over from 1.x,
missing optional dependencies, and what a bare import leaves on disk."""

import builtins
import json
import os
import subprocess
import sys
import types
import warnings
from importlib import metadata

import numpy as np
import pandas as pd
import pytest

import pyphi
from pyphi import examples
from pyphi.exceptions import ConditionallyDependentError
from pyphi.exceptions import InvalidTPM
from pyphi.exceptions import MissingOptionalDependenciesError


def test_version_attribute_matches_the_installed_metadata():
    assert pyphi.__version__ == metadata.version("pyphi")


@pytest.mark.parametrize(
    ("name", "replacement"),
    [("Network", "Substrate"), ("Subsystem", "System"), ("compute", "analyze")],
)
def test_names_from_1x_point_to_their_replacement(name, replacement):
    with pytest.raises(AttributeError, match=replacement) as raised:
        getattr(pyphi, name)
    assert "migration" in str(raised.value)


def test_an_unknown_name_gets_the_ordinary_error():
    with pytest.raises(AttributeError) as raised:
        pyphi.no_such_name  # noqa: B018
    assert "replaced" not in str(raised.value)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [("basic_network", "basic_substrate"), ("basic_subsystem", "basic_system")],
)
def test_example_names_from_1x_point_to_their_replacement(name, replacement):
    with pytest.raises(AttributeError, match=replacement):
        getattr(examples, name)


def test_estimate_analysis_accepts_and_ignores_a_state():
    substrate = examples.basic_substrate()
    assert pyphi.estimate_analysis(substrate, (1, 0, 0)) == pyphi.estimate_analysis(
        substrate
    )


def test_estimate_analysis_rejects_a_subset_passed_as_the_state():
    with pytest.raises(ValueError, match="subset="):
        pyphi.estimate_analysis(examples.basic_substrate(), (0, 1))


def test_unknown_node_labels_are_listed_with_the_known_ones():
    with pytest.raises(ValueError, match=r"\['Z'\].*\['A', 'B', 'C'\]"):
        pyphi.analyze(examples.basic_substrate(), (1, 0, 0), subset=("A", "Z"))


def test_unknown_config_option_links_to_the_published_guide():
    with pytest.raises(Exception, match="https://") as raised:
        pyphi.config.no_such_option = 1
    assert "docs/" not in str(raised.value)


def _without(monkeypatch, package):
    real = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.split(".")[0] == package:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)


def test_missing_pot_names_the_emd_extra(monkeypatch):
    from pyphi.measures import distribution

    _without(monkeypatch, "ot")
    backend = next(
        value()
        for value in vars(distribution).values()
        if isinstance(value, type) and "ot" in vars(value)
    )
    with pytest.raises(ModuleNotFoundError, match=r"pyphi\[emd\]"):
        backend.ot  # noqa: B018


def test_missing_pyarrow_names_the_parquet_extra(monkeypatch):
    from pyphi.serialize import frames

    _without(monkeypatch, "pyarrow")
    with pytest.raises(MissingOptionalDependenciesError, match=r"pyphi\[parquet\]"):
        frames.dataframe_to_schema(pd.DataFrame({"a": [1]}))


def test_pyphi_mcp_without_its_dependency_says_how_to_install_it(monkeypatch, capsys):
    import pyphi.mcp

    monkeypatch.delitem(sys.modules, "pyphi.mcp.server", raising=False)
    _without(monkeypatch, "mcp")
    assert pyphi.mcp.main(["--help"]) == 1
    assert "pyphi[mcp]" in capsys.readouterr().err


def test_importing_pyphi_writes_nothing_to_the_working_directory(tmp_path):
    subprocess.run(
        [sys.executable, "-c", "import pyphi.measures.distribution"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("formalism", ["IIT_3_0", "iit3", "iit_3_0", pyphi.iit3])
def test_a_formalism_is_selected_by_version_name_preset_name_or_preset(formalism):
    from pyphi.conf import presets

    assert presets.canonical(formalism) == "IIT_3_0"


def test_an_unknown_formalism_lists_the_version_names():
    with pytest.raises(ValueError, match="IIT_3_0, IIT_4_0_2023, IIT_4_0_2026"):
        pyphi.analyze(examples.basic_substrate(), (1, 0, 0), formalism="iit5")


def test_sweep_records_the_version_name_for_a_preset_name():
    result = pyphi.sweep(
        examples.basic_substrate(),
        states=[(1, 0, 0)],
        formalisms=["iit4_2023"],
        compute="sia",
        progress=False,
    )
    assert result.df.reset_index()["formalism"].tolist() == ["IIT_4_0_2023"]


def test_uppercase_1x_option_names_work_with_a_warning():
    with pytest.warns(FutureWarning, match="`precision`"):
        assert pyphi.config.precision == pyphi.config.PRECISION
    with (
        pytest.warns(FutureWarning, match="`precision`"),
        pyphi.config.override(PRECISION=6),
    ):
        assert pyphi.config.precision == 6


def test_the_card_says_why_a_deterministic_system_has_zero_system_phi():
    with pyphi.config.override(**pyphi.iit4_2026):
        deterministic = pyphi.analyze(examples.basic_substrate(), (1, 0, 0))
        probabilistic = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1)
        )
    assert deterministic.phi == 0
    assert "no repertoire of alternatives" in str(deterministic)
    assert "Why" not in str(probabilistic)


def test_progress_bars_wait_before_drawing():
    from pyphi._progress import DELAY
    from pyphi._progress import tqdm

    bar = tqdm(range(3), disable=False)
    try:
        assert bar.delay == DELAY > 0
    finally:
        bar.close()


def test_the_welcome_message_is_short():
    # Windows cannot start Python from an empty environment, so drop only the
    # variables that change what is printed.
    silencers = (
        "PYPHI_WELCOME_OFF",
        "PYPHI_AGENT_NOTE_OFF",
        "CLAUDECODE",
        "PYPHI_AGENT",
    )
    environment = {k: v for k, v in os.environ.items() if k not in silencers}
    done = subprocess.run(
        [sys.executable, "-c", "import pyphi"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = done.stderr.strip().splitlines()
    assert 0 < len(lines) <= 6
    assert "10.1371/journal.pcbi.1006343" in done.stderr


# Error messages for the mistakes a new user makes first.


def _tpm(rows):
    return pyphi.Substrate(np.array(rows, dtype=float))


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (np.eye(4) * 0.9, r"row 0 \(input state \(0, 0\)\) sums to 0\.9"),
        (np.eye(4) * 2, r"row 0 .* is 2\.0"),
        ([[0.5, 0.2, 0.2, 0], *np.eye(4)[1:]], r"row 0 .* sums to 0\.9"),
        ([[np.nan, 1, 0, 0], *np.eye(4)[1:]], r"row 0 .* is nan"),
    ],
)
def test_a_state_by_state_tpm_with_a_bad_row_names_the_row(rows, message):
    with pytest.raises(InvalidTPM, match=message):
        _tpm(rows)


def test_a_conditionally_dependent_tpm_names_the_page_that_explains_it():
    # Both units copy one hidden coin: rows are distributions, but the units
    # are not independent given the previous state.
    copied = [[0.5, 0, 0, 0.5]] * 4
    with pytest.raises(ConditionallyDependentError) as raised:
        _tpm(copied)
    assert "theory/conditional-independence" in str(raised.value)
    assert "common cause" in str(raised.value)


def test_an_out_of_range_entry_is_reported_as_entered():
    with pytest.raises(InvalidTPM) as raised:
        _tpm([[0, 0], [0, 1.5], [1, 0], [1, 1]])
    message = str(raised.value)
    assert "1.5" in message
    assert "(1, 0)" in message
    assert "unit 1" in message
    assert "-0.5" not in message


def test_a_nan_entry_is_reported_before_the_row_sum():
    with pytest.raises(InvalidTPM) as raised:
        _tpm([[0, 0], [0, np.nan], [1, 0], [1, 1]])
    message = str(raised.value)
    assert "nan" in message
    assert "sum" not in message
    assert "np.int64" not in message


def test_a_marginal_out_of_range_names_unit_and_state():
    bad = np.full((2, 2, 2), 0.5)
    bad[1, 0] = [1.2, -0.2]
    good = np.full((2, 2, 2), 0.5)
    with pytest.raises(InvalidTPM, match=r"unit 0 at input state \(1, 0\).*1\.2"):
        pyphi.Substrate(marginals=[bad, good])


@pytest.mark.parametrize("call", [pyphi.analyze, pyphi.estimate_analysis])
def test_passing_a_system_where_a_substrate_goes_says_what_to_pass(call):
    with pytest.raises(TypeError, match=r"system\.substrate"):
        call(examples.basic_system(), (1, 0, 0))


def test_passing_an_array_where_a_substrate_goes_says_to_build_one():
    with pytest.raises(TypeError, match=r"pyphi\.Substrate\("):
        pyphi.analyze(np.eye(4), (0, 0))


def test_node_labels_must_be_strings():
    with pytest.raises(ValueError, match="str"):
        pyphi.Substrate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), node_labels=range(2))


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_loading_a_1x_jsonify_file_says_where_it_came_from(tmp_path):
    basic = examples.basic_substrate()
    data = {
        "tpm": np.asarray(basic.tpm).tolist(),
        "cm": basic.cm.tolist(),
        "__class__": "Network",
        "__version__": "1.2.0",
    }
    path = _write(tmp_path / "old.json", data)
    with pytest.raises(ValueError, match=r"1\.x") as raised:
        pyphi.load(path)
    assert "pyphi.Substrate(data['tpm'], cm=data['cm'])" in str(raised.value)
    assert str(path) in "".join(raised.value.__notes__)
    # The rebuild the message suggests gives back the network.
    assert pyphi.Substrate(data["tpm"], cm=data["cm"]) == basic


def test_loading_a_foreign_json_file_says_it_is_not_a_pyphi_file(tmp_path):
    with pytest.raises(ValueError, match="not a file saved by"):
        pyphi.load(_write(tmp_path / "other.json", {"a": 1}))


def test_loading_a_truncated_file_names_the_path(tmp_path):
    path = tmp_path / "cut.json"
    pyphi.save(examples.basic_substrate(), path)
    path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(ValueError, match="truncated") as raised:
        pyphi.load(path)
    assert str(path) in "".join(raised.value.__notes__)


def test_the_card_says_why_a_system_that_is_not_strongly_connected_has_zero_phi():
    with pyphi.config.override(**pyphi.iit4_2026):
        analysis = pyphi.analyze(examples.basic_substrate(), (1, 0, 0), subset=(0, 1))
    assert analysis.sia.current_state == (1, 0)
    card = str(analysis)
    assert "not strongly connected" in card
    assert "None" not in card


def test_an_empty_subset_says_the_system_has_no_units():
    with pyphi.config.override(**pyphi.iit4_2026):
        analysis = pyphi.analyze(examples.basic_substrate(), (1, 0, 0), subset=())
    assert "no units" in str(analysis)


@pytest.mark.parametrize(
    ("state", "error", "message"),
    [
        ((1, 0), ValueError, "background"),
        ("100", TypeError, "state"),
        ({"A": 1, "B": 0, "C": 0}, TypeError, "state"),
        (1, TypeError, "state"),
        (None, TypeError, "state"),
    ],
)
def test_a_malformed_state_says_what_a_state_is(state, error, message):
    with pytest.raises(error, match=message):
        pyphi.analyze(examples.basic_substrate(), state, subset=(0, 1))


@pytest.mark.parametrize(
    ("subset", "error", "message"),
    [
        (0, TypeError, "sequence"),
        ((0, "B"), ValueError, "all labels or all indices"),
        ((0, 0, 1), ValueError, "more than once"),
    ],
)
def test_a_malformed_subset_is_rejected(subset, error, message):
    with pytest.raises(error, match=message):
        pyphi.analyze(examples.basic_substrate(), (1, 0, 0), subset=subset)


def test_a_transition_given_string_states_says_what_a_state_is():
    from pyphi.actual import Transition

    with pytest.raises(TypeError, match="state"):
        Transition(examples.basic_substrate(), "100", "100", (0,), (1,))


@pytest.mark.parametrize("alphabet", [(3, 2), [3, 2]])
def test_alphabet_takes_one_size_per_unit(alphabet):
    substrate = pyphi.Substrate(
        marginals=[np.full((3, 2, 3), 1 / 3), np.full((3, 2, 2), 0.5)],
        alphabet=alphabet,
    )
    assert substrate.state_space == ((0, 1, 2), (0, 1))


def test_an_unknown_unit_function_lists_the_known_ones():
    axis = pyphi.weight_axes("sigmod", np.array([[0, 0.8], [0.8, 0]]), [(0, 1)])
    with pytest.raises(ValueError, match="sigmoid"):
        axis(np.array([1.0]))


def test_a_binary_tpm_converts_to_a_state_by_node_array():
    substrate = examples.basic_substrate()
    array = np.asarray(substrate.tpm)
    assert array.shape == (8, 3)
    np.testing.assert_array_equal(array, substrate.tpm.to_pandas().to_numpy())
    assert pyphi.convert.state_by_node2state_by_state(substrate.tpm).shape == (8, 8)


def test_a_multivalued_tpm_points_to_its_table_forms():
    substrate = pyphi.Substrate(
        marginals=[np.full((3, 2, 3), 1 / 3), np.full((3, 2, 2), 0.5)]
    )
    with pytest.raises(TypeError, match=r"to_pandas\(\)"):
        np.asarray(substrate.tpm)


_NO_IPYWIDGETS = """
import sys, types, warnings
warnings.simplefilter("always")
shell = types.SimpleNamespace(config={"IPKernelApp": {}})
ipython = types.ModuleType("IPython")
ipython.get_ipython = lambda: shell
sys.modules["IPython"] = ipython
sys.modules["ipywidgets"] = None
import %s
"""


def test_importing_pyphi_in_jupyter_without_ipywidgets_does_not_warn():
    environment = {**os.environ, "PYPHI_WELCOME_OFF": "1", "PYPHI_AGENT_NOTE_OFF": "1"}

    def stderr(module):
        return subprocess.run(
            [sys.executable, "-c", _NO_IPYWIDGETS % module],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stderr

    # Control: the simulated notebook without ipywidgets makes this tqdm warn.
    # Environments whose tqdm does not warn there have nothing to suppress.
    if "TqdmWarning" not in stderr("tqdm.auto"):
        pytest.skip("this tqdm does not warn in a notebook without ipywidgets")
    assert "TqdmWarning" not in stderr("pyphi")


def test_importing_pot_does_not_show_its_syntax_warnings(monkeypatch):
    from pyphi.measures import distribution

    real = builtins.__import__

    def compiling(name, *args, **kwargs):
        if name == "ot":
            warnings.warn("invalid escape sequence '\\d'", SyntaxWarning, stacklevel=1)
            return types.ModuleType("ot")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", compiling)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        distribution.OptionalEMD().ot  # noqa: B018
