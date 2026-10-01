"""What someone new to PyPhi 2.0 meets first: names carried over from 1.x,
missing optional dependencies, and what a bare import leaves on disk."""

import builtins
import subprocess
import sys
from importlib import metadata

import pandas as pd
import pytest

import pyphi
from pyphi import examples
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
