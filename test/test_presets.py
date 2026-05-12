"""Tests for canonical IIT settings presets."""

from __future__ import annotations

import pytest

from pyphi import iit3_settings as top_iit3_settings
from pyphi import iit4_2023_settings as top_iit4_2023_settings
from pyphi import iit4_2026_settings as top_iit4_2026_settings
from pyphi.conf import config
from pyphi.conf import iit3_settings
from pyphi.conf import iit4_2023_settings
from pyphi.conf import iit4_2026_settings
from pyphi.conf.formalism import IITConfig


class TestTopLevelReExports:
    def test_iit3_settings_lifted(self):
        assert top_iit3_settings is iit3_settings

    def test_iit4_2023_settings_lifted(self):
        assert top_iit4_2023_settings is iit4_2023_settings

    def test_iit4_2026_settings_lifted(self):
        assert top_iit4_2026_settings is iit4_2026_settings


class TestIIT3Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit3_settings["iit"], IITConfig)

    def test_version(self):
        assert iit3_settings["iit"].version == "IIT_3_0"

    def test_mechanism_phi_measure_is_emd(self):
        assert iit3_settings["iit"].mechanism_phi_measure == "EMD"

    def test_ces_measure_is_emd(self):
        assert iit3_settings["iit"].ces_measure == "EMD"

    def test_mechanism_partition_scheme_is_bi(self):
        assert iit3_settings["iit"].mechanism_partition_scheme == "BI"

    def test_system_partition_scheme_is_directed_bi(self):
        assert iit3_settings["iit"].system_partition_scheme == "DIRECTED_BI"

    def test_no_selfloop_phi(self):
        assert iit3_settings["iit"].single_micro_nodes_with_selfloops_have_phi is False

    def test_purview_tie_resolution(self):
        assert iit3_settings["iit"].purview_tie_resolution == "PHI"

    def test_precision_is_6(self):
        assert iit3_settings["precision"] == 6


class TestIIT4_2023Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit4_2023_settings["iit"], IITConfig)

    def test_version(self):
        assert iit4_2023_settings["iit"].version == "IIT_4_0_2023"

    def test_no_precision_override(self):
        assert "precision" not in iit4_2023_settings

    def test_matches_iitconfig_defaults_apart_from_version(self):
        # The 2023 paper's settings are the IITConfig defaults; the preset
        # exists to give a named entry point and a reset-to-canonical
        # semantics regardless of prior overrides.
        defaults = IITConfig()
        preset = iit4_2023_settings["iit"]
        assert preset.version == "IIT_4_0_2023" == defaults.version
        assert preset.mechanism_phi_measure == defaults.mechanism_phi_measure
        assert preset.system_phi_measure == defaults.system_phi_measure
        assert preset.mechanism_partition_scheme == defaults.mechanism_partition_scheme
        assert preset.system_partition_scheme == defaults.system_partition_scheme


class TestIIT4_2026Settings:
    def test_iit_subnamespace_is_iitconfig(self):
        assert isinstance(iit4_2026_settings["iit"], IITConfig)

    def test_version(self):
        assert iit4_2026_settings["iit"].version == "IIT_4_0_2026"

    def test_system_phi_measure_is_intrinsic_information(self):
        # The 2026 paper caps φ_s by min{i_diff, i_spec} (Eq. 23).
        assert iit4_2026_settings["iit"].system_phi_measure == "INTRINSIC_INFORMATION"

    def test_mechanism_phi_measure_stays_gid(self):
        # 2026 only modifies system-level integration; mechanism phi
        # still uses GID per Eqs. 19-20.
        assert (
            iit4_2026_settings["iit"].mechanism_phi_measure
            == "GENERALIZED_INTRINSIC_DIFFERENCE"
        )

    def test_no_precision_override(self):
        assert "precision" not in iit4_2026_settings


class TestOverrideApplication:
    """Verify each preset round-trips through ``config.override(**preset)``."""

    def test_iit3_applies_and_restores(self):
        original_version = config.formalism.iit.version
        original_precision = config.numerics.precision

        with config.override(**iit3_settings):
            assert config.formalism.iit.version == "IIT_3_0"
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
            assert config.formalism.iit.mechanism_partition_scheme == "BI"
            assert config.formalism.iit.system_partition_scheme == "DIRECTED_BI"
            assert config.numerics.precision == 6

        assert config.formalism.iit.version == original_version
        assert config.numerics.precision == original_precision

    def test_iit4_2023_applies_and_restores(self):
        original = config.formalism.iit

        with config.override(**iit4_2023_settings):
            assert config.formalism.iit.version == "IIT_4_0_2023"
            assert (
                config.formalism.iit.system_phi_measure
                == "GENERALIZED_INTRINSIC_DIFFERENCE"
            )

        assert config.formalism.iit == original

    def test_iit4_2026_applies_and_restores(self):
        original = config.formalism.iit

        with config.override(**iit4_2026_settings):
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
            with config.override(**iit4_2023_settings):
                assert (
                    config.formalism.iit.mechanism_phi_measure
                    == "GENERALIZED_INTRINSIC_DIFFERENCE"
                )

    @pytest.mark.parametrize(
        "preset",
        [iit3_settings, iit4_2023_settings, iit4_2026_settings],
    )
    def test_preset_keys_route_through_override(self, preset):
        # Every key in a preset must be addressable by ``override`` —
        # either as a flat-routable field name or as the ``iit``
        # sub-namespace replacement.
        with config.override(**preset):
            pass
