"""Tests for canonical IIT settings presets."""

from __future__ import annotations

import copy
import dataclasses
import pickle
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
from pyphi.conf import presets
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
    @staticmethod
    def yml() -> dict:
        path = Path(__file__).resolve().parent.parent.parent / "pyphi_config_3.0.yml"
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
            list(iit3["iit"].purview_tie_resolution)
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
        assert iit3["iit"].purview_tie_resolution == ("PHI", "PURVIEW_SIZE")

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

    def test_pins_uncapped_2023_measures_absolutely(self):
        # The preset is an absolute IIT 4.0 (2023) formalism: GID at every
        # level, set explicitly so it never inherits the IITConfig defaults
        # (which now encode the 2026 cap, system_phi_measure=INTRINSIC_INFORMATION).
        # The partition schemes are formalism-independent and match the defaults.
        defaults = IITConfig()
        preset = iit4_2023["iit"]
        assert preset.version == "IIT_4_0_2023"
        assert preset.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert preset.system_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert preset.specification_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
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
        # L1 is an IIT 3.0 measure set on the default 4.0 formalism purely to
        # show the preset resets it; bypass the B13 combination validator.
        with config.override(mechanism_phi_measure="L1", validate_config=False):
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


class TestPresetsAreReadOnly:
    @pytest.mark.parametrize("preset", [iit3, iit4_2023, iit4_2026])
    def test_a_preset_cannot_be_mutated(self, preset):
        with pytest.raises(TypeError):
            preset["precision"] = 2  # type: ignore[index]

    @pytest.mark.parametrize("preset", [iit3, iit4_2023, iit4_2026])
    @pytest.mark.parametrize(
        "copier",
        [
            lambda p: pickle.loads(pickle.dumps(p)),
            copy.deepcopy,
            copy.copy,
        ],
        ids=["pickle", "deepcopy", "copy"],
    )
    def test_a_preset_pickles_and_copies_and_stays_read_only(self, preset, copier):
        duplicate = copier(preset)
        assert duplicate == preset
        assert presets.canonical(duplicate) == preset["iit"].version
        with pytest.raises(TypeError):
            duplicate["precision"] = 2  # type: ignore[index]

    def test_by_name_pickles(self):
        assert pickle.loads(pickle.dumps(presets.by_name)) == presets.by_name

    def test_a_copy_applies_through_override(self):
        with config.override(**{**iit3, "precision": 8}):
            assert config.numerics.precision == 8
            assert config.formalism.iit.version == "IIT_3_0"


class TestFormalismMappings:
    def test_an_unmodified_copy_selects_its_version(self):
        from pyphi.conf import presets

        assert presets.canonical(dict(iit3)) == "IIT_3_0"

    def test_a_modified_preset_is_rejected(self):
        import dataclasses

        from pyphi.conf import presets

        custom = {
            **iit3,
            "iit": dataclasses.replace(iit3["iit"], mechanism_phi_measure="KLD"),
        }
        with pytest.raises(ValueError, match=r"config\.override"):
            presets.canonical(custom)

    def test_analyze_rejects_a_modified_preset(self):
        import pyphi

        custom = {**iit4_2023, "precision": 3}
        with pytest.raises(ValueError, match="differs from the IIT_4_0_2023 preset"):
            pyphi.analyze(pyphi.examples.basic_substrate(), (1, 0, 0), formalism=custom)


class TestPresetsKeepComputationStrategy:
    @pytest.mark.parametrize("preset", [iit3, iit4_2023, iit4_2026])
    def test_override_with_a_preset_keeps_relation_computation(self, preset):
        with config.override(relation_computation="CONCRETE"), config.override(**preset):
            assert config.formalism.iit.relation_computation == "CONCRETE"

    def test_assigning_a_preset_iit_keeps_relation_computation(self):
        with config.override(relation_computation="CONCRETE"):
            config.iit = iit4_2023["iit"]
            assert config.formalism.iit.version == "IIT_4_0_2023"
            assert config.formalism.iit.relation_computation == "CONCRETE"

    @pytest.mark.parametrize(
        "clone", [copy.deepcopy, lambda p: pickle.loads(pickle.dumps(p))]
    )
    def test_a_copied_preset_keeps_relation_computation(self, clone):
        with (
            config.override(relation_computation="CONCRETE"),
            config.override(**clone(iit3)),
        ):
            assert config.formalism.iit.version == "IIT_3_0"
            assert config.formalism.iit.relation_computation == "CONCRETE"

    def test_an_explicit_iit_config_applies_as_given(self):
        explicit = dataclasses.replace(IITConfig(), system_partition_include_total=True)
        with (
            config.override(relation_computation="CONCRETE"),
            config.override(iit=explicit),
        ):
            assert config.formalism.iit.relation_computation == "ANALYTICAL"
        explicit = dataclasses.replace(iit4_2023["iit"], relation_computation="CONCRETE")
        with config.override(iit=explicit):
            assert config.formalism.iit.relation_computation == "CONCRETE"

    def test_a_replaced_iit_config_equal_to_a_preset_applies_as_given(self):
        # The value equals the 2026 preset's IITConfig, but it was built with
        # replace(), so it is an explicit request for closed-form relations.
        with config.override(relation_computation="CONCRETE"):
            wanted = dataclasses.replace(
                config.formalism.iit, relation_computation="ANALYTICAL"
            )
            assert wanted == iit4_2026["iit"]
            config.iit = wanted
            assert config.formalism.iit.relation_computation == "ANALYTICAL"
        with (
            config.override(relation_computation="CONCRETE"),
            config.override(iit=wanted),
        ):
            assert config.formalism.iit.relation_computation == "ANALYTICAL"

    def test_item_and_attribute_writes_of_a_preset_agree(self):
        with config.override(relation_computation="CONCRETE"):
            config["formalism.iit"] = iit4_2023["iit"]
            assert config.formalism.iit.version == "IIT_4_0_2023"
            assert config.formalism.iit.relation_computation == "CONCRETE"

    def test_formalism_argument_keeps_concrete_relations(self):
        import pyphi
        from pyphi.relations import ConcreteRelations

        with config.override(relation_computation="CONCRETE"):
            analysis = pyphi.analyze(
                pyphi.examples.basic_substrate(), (0, 0, 0), formalism="IIT_4_0_2026"
            )
        assert isinstance(analysis.ces.relations, ConcreteRelations)
