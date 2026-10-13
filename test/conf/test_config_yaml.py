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
                alpha_measure: WPMI
            """)
        )
        original = config.formalism.actual_causation.alpha_measure
        try:
            config.load_yaml(str(path))
            assert config.formalism.actual_causation.alpha_measure == "WPMI"
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

    def test_renamed_field_error_names_its_replacement(self, tmp_path):
        """A config written against an earlier name says what to change."""
        from pyphi.conf._field_routing import RENAMED_FIELDS

        for old, new in RENAMED_FIELDS.items():
            path = tmp_path / f"config_{old}.yml"
            path.write_text(f"infrastructure:\n  {old}: 1\n")
            with pytest.raises(ConfigurationError, match=f"It is now '{new[0]}'"):
                config.load_yaml(str(path))

    def test_every_renamed_field_points_at_a_real_one(self):
        """The rename map cannot outlive the fields it points to."""
        from pyphi.conf._field_routing import FIELD_TO_LAYER
        from pyphi.conf._field_routing import RENAMED_FIELDS

        for old, paths in RENAMED_FIELDS.items():
            for new in paths:
                assert new in config, f"{old!r} points at unknown option {new!r}"
            assert old not in FIELD_TO_LAYER, f"{old!r} is still a live field"

    def test_every_1x_option_in_the_migration_table_is_known(self):
        """Each 1.x option in the migration guide's table is a live option
        (spelled in uppercase), renamed, or removed, so its error names it."""
        from pyphi.conf._field_routing import former_option_location

        table = [
            "IIT_VERSION",
            "MEASURE",
            "REPERTOIRE_DISTANCE",
            "REPERTOIRE_DISTANCE_SPECIFICATION",
            "REPERTOIRE_DISTANCE_DIFFERENTIATION",
            "CES_DISTANCE",
            "ACTUAL_CAUSATION_MEASURE",
            "PARTITION_TYPE",
            "SYSTEM_PARTITION_TYPE",
            "SYSTEM_PARTITION_INCLUDE_COMPLETE",
            "SYSTEM_CUTS",
            "DISTINCTION_PHI_NORMALIZATION",
            "RELATION_COMPUTATION",
            "STATE_TIE_RESOLUTION",
            "MIP_TIE_RESOLUTION",
            "PURVIEW_TIE_RESOLUTION",
            "PICK_SMALLEST_PURVIEW",
            "ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS",
            "SHORTCIRCUIT_SIA",
            "SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI",
            "PRECISION",
            "PARALLEL",
            "PARALLEL_WORKERS",
            "PARALLEL_BACKEND",
            "PARALLEL_COMPLEX_EVALUATION",
            "PARALLEL_PURVIEW_EVALUATION",
            "PARALLEL_MECHANISM_PARTITION_EVALUATION",
            "PARALLEL_RELATION_EVALUATION",
            "PARALLEL_CUT_EVALUATION",
            "PARALLEL_CONCEPT_EVALUATION",
            "MAXIMUM_CACHE_MEMORY_PERCENTAGE",
            "CACHE_REPERTOIRES",
            "CACHE_POTENTIAL_PURVIEWS",
            "CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA",
            "REDIS_CACHE",
            "REDIS_CONFIG",
            "RAY_CONFIG",
            "LOG_FILE",
            "LOG_FILE_LEVEL",
            "LOG_STDOUT_LEVEL",
            "PROGRESS_BARS",
            "WELCOME_OFF",
            "REPR_VERBOSITY",
            "PRINT_FRACTIONS",
            "LABEL_SEPARATOR",
            "VALIDATE_SUBSYSTEM_STATES",
            "VALIDATE_CONDITIONAL_INDEPENDENCE",
            "VALIDATE_JSON_VERSION",
        ]
        unknown = [name for name in table if former_option_location(name) is None]
        assert unknown == []

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

    def test_a_version_alone_names_the_file_and_the_preset(self, tmp_path):
        (tmp_path / "pyphi_config.yml").write_text(
            "formalism:\n  iit:\n    version: IIT_3_0\n"
        )
        result = self._import_pyphi(tmp_path)
        assert result.returncode != 0
        assert str(tmp_path / "pyphi_config.yml") in result.stderr
        assert "pyphi_config_3.0.yml" in result.stderr

    def test_an_unknown_option_names_the_file(self, tmp_path):
        (tmp_path / "pyphi_config.yml").write_text(
            "formalism:\n  iit:\n    mechanism_phi_measur: EMD\n"
        )
        result = self._import_pyphi(tmp_path)
        assert result.returncode != 0
        assert str(tmp_path / "pyphi_config.yml") in result.stderr
        assert "Did you mean 'mechanism_phi_measure'?" in result.stderr

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

    @pytest.mark.parametrize(
        "text",
        ["formalism:\n  iit: 5\n", "numerics: 3\n", "formalism: [1]\n"],
    )
    def test_a_section_that_is_not_a_mapping_is_rejected(self, tmp_path, text):
        path = tmp_path / "config.yml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="must map option names"):
            config.load_yaml(str(path))

    def test_a_misspelt_nested_option_suggests_the_name(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("formalism:\n  iit:\n    mechanism_phi_measur: EMD\n")
        with pytest.raises(ConfigurationError, match="mechanism_phi_measure"):
            config.load_yaml(str(path))

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


class TestShippedConfigFiles:
    def test_iit3_reference_yaml_reproduces_preset(self):
        """The shipped ``pyphi_config_3.0.yml`` loads and mirrors the
        ``IIT_3_0`` preset exactly."""
        from pathlib import Path

        from pyphi.conf import presets

        path = Path(__file__).parents[2] / "pyphi_config_3.0.yml"
        before = config.snapshot()
        try:
            config.load_yaml(str(path))
            for key, want in presets.by_name["IIT_3_0"].items():
                assert getattr(config, key) == want, key
        finally:
            config.install_snapshot(before)


class TestEmptyLayerSections:
    def test_empty_layer_section_loads(self, tmp_path):
        """A layer key with nothing under it (YAML None) is an empty dict,
        not an AttributeError."""
        path = tmp_path / "config.yml"
        path.write_text("---\nnumerics:\nformalism:\n  iit:\ninfrastructure:\n")
        before = config.snapshot()
        try:
            config.load_yaml(str(path))  # must not raise
        finally:
            config.install_snapshot(before)

    def test_unknown_option_error_names_migration_guide(self):
        with pytest.raises(ConfigurationError, match="migration"):
            config.not_a_real_option = 1


class TestRenamedFieldHint:
    def test_unknown_field_matching_a_renamed_field_names_its_replacement(self):
        """The error for a renamed option names the replacement, consulting
        RENAMED_FIELDS rather than only pointing at the migration guide."""
        with pytest.raises(ConfigurationError, match="memory_ceiling_bytes"):
            config.maximum_cache_memory_bytes = 0
