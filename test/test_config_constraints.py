"""B13 — eager config-combination validation.

Cross-field config constraints (``pyphi/conf/constraints.py``) reject
silently-wrong combinations of individually-valid options at configuration
time, rather than at compute time or not at all. These tests pin:

  - every shipped preset passes (the validator must never reject a canonical
    config);
  - known-wrong measure/version combinations raise ``ConfigurationError`` with
    a message naming both conflicting fields and a fix;
  - the confirmed-*valid* ``IIT_4_0_2023`` + ``INTRINSIC_INFORMATION`` case is
    NOT rejected (the Eq. 23 cap follows the measure, not the version);
  - ``validate_config=False`` opts out;
  - a rejected override/load does not corrupt global config state;
  - an enumeration over each version's ``compatible_measures`` is classified
    consistently with the reactive ``check_measure_compatible`` boundary.
"""

from __future__ import annotations

import textwrap

import pytest

from pyphi.conf import ConfigurationError
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.formalism.base import FORMALISM_REGISTRY
from pyphi.partition import system_partition_types

# Versions whose formalism consults system_phi_measure (the 4.0 family).
_FOUR_OH = ["IIT_4_0_2023", "IIT_4_0_2026"]

# Each IIT version's canonical preset (sets a system_partition_scheme the
# formalism accepts).
_VERSION_PRESETS = {
    "IIT_3_0": "iit3",
    "IIT_4_0_2023": "iit4_2023",
    "IIT_4_0_2026": "iit4_2026",
}


class TestPresetsPass:
    @pytest.mark.parametrize("preset_name", ["iit3", "iit4_2023", "iit4_2026"])
    def test_canonical_preset_validates(self, preset_name: str) -> None:
        preset = getattr(presets, preset_name)
        with config.override(**preset):
            pass  # no ConfigurationError


class TestIncompatibleCombosRejected:
    def test_iit3_with_intrinsic_information_mechanism(self) -> None:
        """The roadmap's headline example: IIT 3.0 + an IIT-4.0-only measure."""
        with (
            pytest.raises(ConfigurationError) as exc,
            config.override(
                **presets.iit3,
                **{"iit.mechanism_phi_measure": "INTRINSIC_INFORMATION"},
            ),
        ):
            pass
        message = str(exc.value)
        # Names both conflicting fields and offers a fix.
        assert "mechanism_phi_measure" in message
        assert "version" in message
        assert "INTRINSIC_INFORMATION" in message
        assert "Fix" in message

    def test_iit4_with_emd_mechanism(self) -> None:
        with (
            pytest.raises(ConfigurationError, match="mechanism_phi_measure"),
            config.override(**{"iit.mechanism_phi_measure": "EMD"}),
        ):
            pass

    def test_iit4_with_emd_system_measure(self) -> None:
        with (
            pytest.raises(ConfigurationError, match="system_phi_measure"),
            config.override(**{"iit.system_phi_measure": "EMD"}),
        ):
            pass

    def test_unregistered_version(self) -> None:
        with (
            pytest.raises(ConfigurationError, match="not a registered"),
            config.override(**{"iit.version": "IIT_9_9_BOGUS"}),
        ):
            pass


class TestConfirmedValidCombos:
    def test_iit4_2023_with_intrinsic_information_system_is_allowed(self) -> None:
        """The cap is keyed on the measure (``applies_ii_cap``), not the version,
        so IIT_4_0_2023 + INTRINSIC_INFORMATION correctly applies the cap and is
        a valid (if redundant) configuration — it must NOT be rejected."""
        with config.override(**{"iit.system_phi_measure": "INTRINSIC_INFORMATION"}):
            pass

    def test_iit3_system_measure_left_at_default_is_allowed(self) -> None:
        """IIT 3.0 never consults system_phi_measure, so its (4.0-default) value
        is not flagged."""
        with config.override(**presets.iit3):
            assert (
                config.formalism.iit.system_phi_measure
                == "GENERALIZED_INTRINSIC_DIFFERENCE"
            )


class TestOptOut:
    def test_validate_config_false_bypasses(self) -> None:
        with config.override(
            validate_config=False, **{"iit.mechanism_phi_measure": "EMD"}
        ):
            assert config.formalism.iit.mechanism_phi_measure == "EMD"

    def test_validate_config_default_true(self) -> None:
        assert config.infrastructure.validate_config is True


class TestNoStateCorruption:
    def test_rejected_override_restores_state(self) -> None:
        before = config.formalism.iit.mechanism_phi_measure
        with (
            pytest.raises(ConfigurationError),
            config.override(**{"iit.mechanism_phi_measure": "EMD"}),
        ):
            pass
        assert config.formalism.iit.mechanism_phi_measure == before

    def test_rejected_yaml_load_restores_state(self, tmp_path) -> None:
        before = config.snapshot()
        path = tmp_path / "bad.yml"
        path.write_text(
            textwrap.dedent("""\
            ---
            formalism:
              iit:
                mechanism_phi_measure: EMD
            """)
        )
        try:
            with pytest.raises(ConfigurationError):
                config.load_yaml(str(path))
            assert (
                config.formalism.iit.mechanism_phi_measure
                == before.formalism.iit.mechanism_phi_measure
            )
        finally:
            config.install_snapshot(before)


class TestEnumerationConsistency:
    """Every (version, mechanism measure) pair is classified consistently with
    the version's ``compatible_measures`` set."""

    @pytest.mark.parametrize("version", ["IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"])
    def test_every_compatible_mechanism_measure_passes(self, version: str) -> None:
        # Base off the version's preset so the system_partition_scheme is one the
        # formalism accepts (otherwise IIT 3.0 + the default 4.0 scheme is itself a
        # rejected combination, unrelated to the measure under test).
        preset = getattr(presets, _VERSION_PRESETS[version])
        compatible = FORMALISM_REGISTRY[version].compatible_measures
        for measure in compatible:
            with config.override(**preset, **{"iit.mechanism_phi_measure": measure}):
                assert config.formalism.iit.mechanism_phi_measure == measure

    @pytest.mark.parametrize(
        ("version", "incompatible"),
        [
            ("IIT_3_0", "GENERALIZED_INTRINSIC_DIFFERENCE"),
            ("IIT_3_0", "INTRINSIC_INFORMATION"),
            ("IIT_4_0_2023", "EMD"),
            ("IIT_4_0_2023", "L1"),
            ("IIT_4_0_2026", "EMD"),
        ],
    )
    def test_incompatible_mechanism_measure_rejected(
        self, version: str, incompatible: str
    ) -> None:
        compatible = FORMALISM_REGISTRY[version].compatible_measures
        assert incompatible not in compatible  # sanity: truly incompatible
        with (
            pytest.raises(ConfigurationError),
            config.override(
                **{"iit.version": version, "iit.mechanism_phi_measure": incompatible}
            ),
        ):
            pass


class TestSystemSchemeSingleSourceOfTruth:
    def test_iit3_formalism_declares_restricted_scheme_set(self) -> None:
        formalism = FORMALISM_REGISTRY["IIT_3_0"]
        assert formalism.compatible_system_partition_schemes == frozenset(
            {"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"}
        )

    @pytest.mark.parametrize("version", _FOUR_OH)
    def test_iit4_formalism_leaves_scheme_open(self, version: str) -> None:
        assert FORMALISM_REGISTRY[version].compatible_system_partition_schemes is None

    def test_reactive_raise_reads_the_attribute(self) -> None:
        """sia_partitions() rejects an out-of-set scheme under IIT 3.0, using the
        same set the formalism declares (validation off, so the eager constraint
        is not what fires)."""
        from pyphi import examples

        out_of_set = "DIRECTED_SET_PARTITION"
        compatible = FORMALISM_REGISTRY["IIT_3_0"].compatible_system_partition_schemes
        assert compatible is not None and out_of_set not in compatible
        with (
            config.override(
                **presets.iit3,
                validate_config=False,
                **{"iit.system_partition_scheme": out_of_set},
            ),
            pytest.raises(ValueError, match="system partition scheme"),
        ):
            examples.basic_system().sia()


class TestSystemSchemeConstraint:
    def test_iit3_with_set_partition_scheme_rejected(self) -> None:
        with (
            pytest.raises(ConfigurationError) as exc,
            config.override(
                **presets.iit3,
                **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"},
            ),
        ):
            pass
        message = str(exc.value)
        assert "system_partition_scheme" in message
        assert "version" in message
        assert "DIRECTED_SET_PARTITION" in message
        assert "Fix" in message

    @pytest.mark.parametrize(
        "scheme", ["DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"]
    )
    def test_iit3_with_valid_scheme_passes(self, scheme: str) -> None:
        with config.override(**presets.iit3, **{"iit.system_partition_scheme": scheme}):
            assert config.formalism.iit.system_partition_scheme == scheme

    @pytest.mark.parametrize("version", _FOUR_OH)
    def test_iit4_accepts_every_registered_scheme(self, version: str) -> None:
        for scheme in system_partition_types.store:
            with config.override(
                **{"iit.version": version, "iit.system_partition_scheme": scheme}
            ):
                assert config.formalism.iit.system_partition_scheme == scheme

    def test_validate_config_false_bypasses_scheme_constraint(self) -> None:
        with config.override(
            **presets.iit3,
            validate_config=False,
            **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"},
        ):
            assert (
                config.formalism.iit.system_partition_scheme == "DIRECTED_SET_PARTITION"
            )

    def test_rejected_scheme_override_restores_state(self) -> None:
        with config.override(**presets.iit3):
            before = config.formalism.iit.system_partition_scheme
            with (
                pytest.raises(ConfigurationError),
                config.override(
                    **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"}
                ),
            ):
                pass
            assert config.formalism.iit.system_partition_scheme == before


class TestSystemSchemeEnumerationConsistency:
    """The eager constraint's accept/reject for IIT 3.0 matches whether a real
    SIA computes vs raises, for every registered system scheme."""

    def test_iit3_classification_matches_sia_behavior(self) -> None:
        from pyphi import examples

        for scheme in sorted(system_partition_types.store):
            eager_rejected = False
            try:
                with config.override(
                    **presets.iit3, **{"iit.system_partition_scheme": scheme}
                ):
                    pass
            except ConfigurationError:
                eager_rejected = True

            sia_raised = False
            try:
                with config.override(
                    **presets.iit3,
                    validate_config=False,
                    **{"iit.system_partition_scheme": scheme},
                ):
                    examples.basic_system().sia()
            except ValueError:
                sia_raised = True

            assert eager_rejected == sia_raised, scheme
