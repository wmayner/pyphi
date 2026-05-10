"""Tests for the layered (2.0) YAML loader and writer."""

from __future__ import annotations

import textwrap

import pytest

from pyphi.conf import config
from pyphi.conf._field_routing import ConfigurationError


class TestNestedYAMLLoader:
    def test_load_nested_overrides(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            formalism:
              iit:
                mechanism_phi_measure: EMD
            numerics:
              precision: 7
            """)
        )
        original_metric = config.formalism.iit.mechanism_phi_measure
        original_precision = config.numerics.precision
        try:
            config.load_yaml(str(path))
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
            assert config.numerics.precision == 7
        finally:
            config.mechanism_phi_measure = original_metric
            config.precision = original_precision

    def test_load_actual_causation_subnamespace(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            formalism:
              actual_causation:
                alpha_measure: KLD
            """)
        )
        original = config.formalism.actual_causation.alpha_measure
        try:
            config.load_yaml(str(path))
            assert config.formalism.actual_causation.alpha_measure == "KLD"
        finally:
            config.alpha_measure = original

    def test_old_flat_format_raises_with_rename_map(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            PRECISION: 13
            PARALLEL: false
            """)
        )
        with pytest.raises(ConfigurationError, match="rename map"):
            config.load_yaml(str(path))

    def test_unknown_top_level_key_raises(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("nonexistent: 1\n")
        with pytest.raises(ConfigurationError, match="Unknown top-level YAML key"):
            config.load_yaml(str(path))


class TestNestedYAMLWriter:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "config.yml"
        config.to_yaml(str(path))
        # Read back and verify the structure has the three layer keys.
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        assert set(data) == {"formalism", "infrastructure", "numerics"}
        assert data["numerics"]["precision"] == config.numerics.precision
        assert (
            data["formalism"]["iit"]["mechanism_phi_measure"]
            == config.formalism.iit.mechanism_phi_measure
        )
        assert (
            data["formalism"]["actual_causation"]["alpha_measure"]
            == config.formalism.actual_causation.alpha_measure
        )
