"""Tests for canonical IIT settings presets."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pyphi import iit3 as top_iit3
from pyphi import iit4_2023 as top_iit4_2023
from pyphi import iit4_2026 as top_iit4_2026
from pyphi.conf import config
from pyphi.conf import iit3
from pyphi.conf import iit4_2023
from pyphi.conf import iit4_2026
from pyphi.conf.formalism import IITConfig


class TestTopLevelReExports:
    def test_iit3_lifted(self):
        assert top_iit3 is iit3

    def test_iit4_2023_lifted(self):
        assert top_iit4_2023 is iit4_2023

    def test_iit4_2026_lifted(self):
        assert top_iit4_2026 is iit4_2026


class TestIIT3MatchesYamlReference:
    """``presets.iit3`` should encode the same IIT 3.0 fields as the
    library's reference ``pyphi_config_3.0.yml``.
    """

    @pytest.fixture(scope="class")
    def yml(self) -> dict:
        path = Path(__file__).resolve().parent.parent / "pyphi_config_3.0.yml"
        with path.open() as f:
            return yaml.safe_load(f)

    def test_mechanism_phi_measure(self, yml):
        assert (
            iit3["iit"].mechanism_phi_measure
            == yml["formalism"]["iit"]["mechanism_phi_measure"]
        )

    def test_ces_measure(self, yml):
        assert iit3["iit"].ces_measure == yml["formalism"]["iit"]["ces_measure"]

    def test_mechanism_partition_scheme(self, yml):
        assert (
            iit3["iit"].mechanism_partition_scheme
            == yml["formalism"]["iit"]["mechanism_partition_scheme"]
        )

    def test_system_partition_scheme(self, yml):
        assert (
            iit3["iit"].system_partition_scheme
            == yml["formalism"]["iit"]["system_partition_scheme"]
        )

    def test_purview_tie_resolution(self, yml):
        assert (
            iit3["iit"].purview_tie_resolution
            == yml["formalism"]["iit"]["purview_tie_resolution"]
        )

    def test_single_micro_nodes_with_selfloops_have_phi(self, yml):
        assert (
            iit3["iit"].single_micro_nodes_with_selfloops_have_phi
            == yml["formalism"]["iit"]["single_micro_nodes_with_selfloops_have_phi"]
        )

    def test_assume_partitions_cannot_create_new_concepts(self, yml):
        assert (
            iit3["iit"].assume_partitions_cannot_create_new_concepts
            == yml["formalism"]["iit"]["assume_partitions_cannot_create_new_concepts"]
        )

    def test_actual_causation_alpha_measure(self, yml):
        assert (
            iit3["actual_causation"].alpha_measure
            == yml["formalism"]["actual_causation"]["alpha_measure"]
        )

    def test_precision(self, yml):
        assert iit3["precision"] == yml["numerics"]["precision"]


class TestIIT3Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit3["iit"], IITConfig)

    def test_version(self):
        assert iit3["iit"].version == "IIT_3_0"

    def test_mechanism_phi_measure_is_emd(self):
        assert iit3["iit"].mechanism_phi_measure == "EMD"

    def test_ces_measure_is_emd(self):
        assert iit3["iit"].ces_measure == "EMD"

    def test_mechanism_partition_scheme_is_joint_bipartition(self):
        assert iit3["iit"].mechanism_partition_scheme == "JOINT_BIPARTITION"

    def test_system_partition_scheme_is_directed_bipartition(self):
        assert iit3["iit"].system_partition_scheme == "DIRECTED_BIPARTITION"

    def test_no_selfloop_phi(self):
        assert iit3["iit"].single_micro_nodes_with_selfloops_have_phi is False

    def test_purview_tie_resolution(self):
        assert iit3["iit"].purview_tie_resolution == ["PHI", "PURVIEW_SIZE"]

    def test_assume_partitions_cannot_create_new_concepts_is_false(self):
        assert iit3["iit"].assume_partitions_cannot_create_new_concepts is False

    def test_actual_causation_alpha_measure_is_pmi(self):
        assert iit3["actual_causation"].alpha_measure == "PMI"

    def test_precision_is_6(self):
        assert iit3["precision"] == 6


class TestIIT4_2023Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit4_2023["iit"], IITConfig)

    def test_version(self):
        assert iit4_2023["iit"].version == "IIT_4_0_2023"

    def test_no_precision_override(self):
        assert "precision" not in iit4_2023

    def test_matches_iitconfig_defaults_apart_from_version(self):
        # The 2023 paper's settings are the IITConfig defaults; the preset
        # exists to give a named entry point and a reset-to-canonical
        # semantics regardless of prior overrides.
        defaults = IITConfig()
        preset = iit4_2023["iit"]
        assert preset.version == "IIT_4_0_2023" == defaults.version
        assert preset.mechanism_phi_measure == defaults.mechanism_phi_measure
        assert preset.system_phi_measure == defaults.system_phi_measure
        assert preset.mechanism_partition_scheme == defaults.mechanism_partition_scheme
        assert preset.system_partition_scheme == defaults.system_partition_scheme


class TestIIT4_2026Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit4_2026["iit"], IITConfig)

    def test_version(self):
        assert iit4_2026["iit"].version == "IIT_4_0_2026"

    def test_system_phi_measure_is_intrinsic_information(self):
        # The 2026 paper caps φ_s by min{i_diff, i_spec} (Eq. 23).
        assert iit4_2026["iit"].system_phi_measure == "INTRINSIC_INFORMATION"

    def test_mechanism_phi_measure_stays_gid(self):
        # 2026 only modifies system-level integration; mechanism phi
        # still uses GID per Eqs. 19-20.
        assert (
            iit4_2026["iit"].mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        )

    def test_no_precision_override(self):
        assert "precision" not in iit4_2026


class TestOverrideApplication:
    """Verify each preset round-trips through ``config.override(**preset)``."""

    def test_iit3_applies_and_restores(self):
        original_version = config.formalism.iit.version
        original_precision = config.numerics.precision

        with config.override(**iit3):
            assert config.formalism.iit.version == "IIT_3_0"
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
            assert config.formalism.iit.mechanism_partition_scheme == "JOINT_BIPARTITION"
            assert config.formalism.iit.system_partition_scheme == "DIRECTED_BIPARTITION"
            assert config.numerics.precision == 6

        assert config.formalism.iit.version == original_version
        assert config.numerics.precision == original_precision

    def test_iit4_2023_applies_and_restores(self):
        original = config.formalism.iit

        with config.override(**iit4_2023):
            assert config.formalism.iit.version == "IIT_4_0_2023"
            assert (
                config.formalism.iit.system_phi_measure
                == "GENERALIZED_INTRINSIC_DIFFERENCE"
            )

        assert config.formalism.iit == original

    def test_iit4_2026_applies_and_restores(self):
        original = config.formalism.iit

        with config.override(**iit4_2026):
            assert config.formalism.iit.version == "IIT_4_0_2026"
            assert config.formalism.iit.system_phi_measure == "INTRINSIC_INFORMATION"
            assert (
                config.formalism.iit.mechanism_phi_measure
                == "GENERALIZED_INTRINSIC_DIFFERENCE"
            )

        assert config.formalism.iit == original

    def test_preset_resets_prior_iit_overrides(self):
        # A preset replaces the whole IIT sub-namespace, so prior field
        # mutations inside the same override scope are not preserved
        # under a second preset application.
        with config.override(mechanism_phi_measure="L1"):
            assert config.formalism.iit.mechanism_phi_measure == "L1"
            with config.override(**iit4_2023):
                assert (
                    config.formalism.iit.mechanism_phi_measure
                    == "GENERALIZED_INTRINSIC_DIFFERENCE"
                )

    @pytest.mark.parametrize(
        "preset",
        [iit3, iit4_2023, iit4_2026],
    )
    def test_preset_keys_route_through_override(self, preset):
        # Every key in a preset must be addressable by ``override`` —
        # either as a flat-routable field name or as the ``iit``
        # sub-namespace replacement.
        with config.override(**preset):
            pass
