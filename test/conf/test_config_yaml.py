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
            infrastructure:
              validate_config: false
            """)
        )
        original_measure = config.formalism.iit.mechanism_phi_measure
        original_precision = config.numerics.precision
        try:
            config.load_yaml(str(path))
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
            assert config.numerics.precision == 7
        finally:
            config.mechanism_phi_measure = original_measure
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

    def test_load_colliding_subnamespace_fields_route_by_nesting(self, tmp_path):
        # ``version`` and ``mechanism_partition_scheme`` exist in both the
        # iit and actual_causation sub-namespaces. A nested YAML must route
        # each to the sub-namespace it is nested under, not raise on the
        # ambiguous bare name.
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            formalism:
              iit:
                version: IIT_3_0
                mechanism_partition_scheme: JOINT_BIPARTITION
            infrastructure:
              validate_config: false
            """)
        )
        original = config.snapshot()
        try:
            config.load_yaml(str(path))
            assert config.formalism.iit.version == "IIT_3_0"
            assert config.formalism.iit.mechanism_partition_scheme == "JOINT_BIPARTITION"
            # The AC sub-namespace keeps its default version.
            assert config.formalism.actual_causation.version == "AC_2019"
        finally:
            config.install_snapshot(original)

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


class TestImportTimeValidation:
    """The auto-loaded ``pyphi_config.yml`` gets the same eager constraint
    check as ``override``/``load_yaml`` (subprocess: import-time behavior)."""

    def _import_pyphi(self, cwd):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-c", "import pyphi"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120,
            check=False,  # tests assert on returncode explicitly
        )

    def test_invalid_yaml_combo_fails_at_import(self, tmp_path):
        (tmp_path / "pyphi_config.yml").write_text(
            textwrap.dedent("""\
            ---
            formalism:
              iit:
                version: IIT_4_0_2023
                mechanism_phi_measure: EMD
            infrastructure:
              welcome_off: true
            """)
        )
        result = self._import_pyphi(tmp_path)
        assert result.returncode != 0
        assert "ConfigurationError" in result.stderr

    def test_valid_yaml_imports_cleanly(self, tmp_path):
        (tmp_path / "pyphi_config.yml").write_text(
            textwrap.dedent("""\
            ---
            numerics:
              precision: 10
            infrastructure:
              welcome_off: true
            """)
        )
        result = self._import_pyphi(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_validate_config_opt_out_respected(self, tmp_path):
        (tmp_path / "pyphi_config.yml").write_text(
            textwrap.dedent("""\
            ---
            formalism:
              iit:
                version: IIT_4_0_2023
                mechanism_phi_measure: EMD
            infrastructure:
              validate_config: false
              welcome_off: true
            """)
        )
        result = self._import_pyphi(tmp_path)
        assert result.returncode == 0, result.stderr


class TestSectionRouting:
    def test_misfiled_field_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            numerics:
              parallel: true
            """)
        )
        before = config.infrastructure.parallel
        with pytest.raises(ConfigurationError, match="belongs to"):
            config.load_yaml(str(path))
        assert config.infrastructure.parallel == before

    def test_stray_formalism_key_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            formalism:
              parallel: true
            """)
        )
        with pytest.raises(ConfigurationError, match="unrecognized keys"):
            config.load_yaml(str(path))
