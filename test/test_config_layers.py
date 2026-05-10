"""Tests for the layered config dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields
from dataclasses import replace

import pytest

from pyphi.conf import config
from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf._field_routing import colliding_formalism_fields
from pyphi.conf._global import _GlobalConfig
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot


class TestNumericsConfig:
    def test_default_precision_is_13(self):
        cfg = NumericsConfig()
        assert cfg.precision == 13

    def test_explicit_precision(self):
        cfg = NumericsConfig(precision=6)
        assert cfg.precision == 6

    def test_is_frozen(self):
        cfg = NumericsConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.precision = 6  # type: ignore[misc]

    def test_equality_by_value(self):
        assert NumericsConfig(precision=13) == NumericsConfig(precision=13)
        assert NumericsConfig(precision=13) != NumericsConfig(precision=6)

    def test_hashable(self):
        assert hash(NumericsConfig(precision=13)) == hash(NumericsConfig(precision=13))


class TestIITConfig:
    def test_defaults(self):
        cfg = IITConfig()
        assert cfg.version == "IIT_4_0_2023"
        assert cfg.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert cfg.mechanism_partition_scheme == "ALL"
        assert cfg.system_partition_scheme == "SET_UNI/BI"
        assert cfg.shortcircuit_sia is True

    def test_is_frozen(self):
        cfg = IITConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.version = "IIT_3_0"  # type: ignore[misc]

    def test_replace_returns_new_instance(self):
        a = IITConfig()
        b = replace(a, mechanism_phi_measure="EMD")
        assert a.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert b.mechanism_phi_measure == "EMD"

    def test_mip_tie_resolution_default_is_list(self):
        cfg = IITConfig()
        assert cfg.mip_tie_resolution == ["NORMALIZED_PHI", "NEGATIVE_PHI"]

    def test_mip_tie_resolution_each_instance_independent(self):
        a = IITConfig()
        b = IITConfig()
        assert a.mip_tie_resolution is not b.mip_tie_resolution


class TestActualCausationConfig:
    def test_defaults_match_paper(self):
        cfg = ActualCausationConfig()
        assert cfg.alpha_measure == "PMI"
        assert cfg.mechanism_partition_scheme == "ALL"
        assert cfg.partitioned_repertoire_scheme == "PRODUCT"
        assert cfg.background_scheme == "UNIFORM"
        assert cfg.alpha_aggregation == "SUBTRACTIVE"

    def test_is_frozen(self):
        cfg = ActualCausationConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.alpha_measure = "KLD"  # type: ignore[misc]

    def test_invalid_partitioned_repertoire_scheme_raises(self):
        with pytest.raises(ValueError, match="partitioned_repertoire_scheme"):
            ActualCausationConfig(partitioned_repertoire_scheme="BOGUS")

    def test_invalid_background_scheme_raises(self):
        with pytest.raises(ValueError, match="background_scheme"):
            ActualCausationConfig(background_scheme="BOGUS")

    def test_invalid_alpha_aggregation_raises(self):
        with pytest.raises(ValueError, match="alpha_aggregation"):
            ActualCausationConfig(alpha_aggregation="BOGUS")


class TestFormalismConfig:
    def test_holds_two_subnamespaces(self):
        cfg = FormalismConfig()
        assert isinstance(cfg.iit, IITConfig)
        assert isinstance(cfg.actual_causation, ActualCausationConfig)

    def test_iit_defaults_visible_through_formalism(self):
        cfg = FormalismConfig()
        assert cfg.iit.version == "IIT_4_0_2023"
        assert cfg.iit.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"

    def test_ac_defaults_visible_through_formalism(self):
        cfg = FormalismConfig()
        assert cfg.actual_causation.alpha_measure == "PMI"

    def test_is_frozen(self):
        cfg = FormalismConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.iit = IITConfig()  # type: ignore[misc]

    def test_replace_swaps_iit(self):
        a = FormalismConfig()
        new_iit = replace(a.iit, mechanism_phi_measure="EMD")
        b = replace(a, iit=new_iit)
        assert a.iit.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert b.iit.mechanism_phi_measure == "EMD"

    def test_as_kwargs_flattens_iit_and_ac_subnamespaces(self):
        cfg = FormalismConfig(iit=IITConfig(mechanism_phi_measure="EMD"))
        kw = cfg.as_kwargs()
        assert kw["mechanism_phi_measure"] == "EMD"
        assert kw["alpha_measure"] == "PMI"
        assert kw["partitioned_repertoire_scheme"] == "PRODUCT"

    def test_as_kwargs_excludes_colliding_fields(self):
        cfg = FormalismConfig()
        kw = cfg.as_kwargs()
        for collider in colliding_formalism_fields():
            assert collider not in kw


class TestInfrastructureConfig:
    def test_defaults_match_legacy(self):
        cfg = InfrastructureConfig()
        assert cfg.parallel is False
        assert cfg.parallel_workers == -1
        assert cfg.parallel_backend == "local"
        assert cfg.cache_repertoires is True
        assert cfg.cache_potential_purviews is True
        assert cfg.clear_system_caches_after_computing_sia is False
        assert cfg.maximum_cache_memory_percentage == 50
        assert cfg.log_file_level == "INFO"
        assert cfg.log_stdout_level == "WARNING"
        assert cfg.progress_bars is True
        assert cfg.repr_verbosity == 2
        assert cfg.print_fractions is True
        assert cfg.label_separator == ""
        assert cfg.welcome_off is False
        assert cfg.validate_system_states is True
        assert cfg.validate_conditional_independence is True
        assert cfg.validate_json_version is True

    def test_parallel_evaluation_dict_has_expected_keys(self):
        cfg = InfrastructureConfig()
        assert set(cfg.parallel_complex_evaluation.keys()) == {
            "parallel",
            "sequential_threshold",
            "chunksize",
            "progress",
        }

    def test_is_frozen(self):
        cfg = InfrastructureConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.parallel = True  # type: ignore[misc]


class TestConfigSnapshot:
    def test_construction(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        assert snap.numerics.precision == 13
        assert snap.formalism.iit.version == "IIT_4_0_2023"

    def test_is_frozen(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        with pytest.raises(FrozenInstanceError):
            snap.numerics = NumericsConfig(precision=6)  # type: ignore[misc]

    def test_diff_finds_differences(self):
        a = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(precision=13),
        )
        b = ConfigSnapshot(
            formalism=FormalismConfig(iit=IITConfig(mechanism_phi_measure="EMD")),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(precision=6),
        )
        diff = a.diff(b)
        assert diff == {
            "formalism.iit.mechanism_phi_measure": (
                "GENERALIZED_INTRINSIC_DIFFERENCE",
                "EMD",
            ),
            "numerics.precision": (13, 6),
        }

    def test_diff_empty_when_equal(self):
        a = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        b = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        assert a.diff(b) == {}

    def test_as_kwargs_returns_flat_dict(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(iit=IITConfig(mechanism_phi_measure="EMD")),
            infrastructure=InfrastructureConfig(parallel=True),
            numerics=NumericsConfig(precision=6),
        )
        kw = snap.as_kwargs()
        assert kw["mechanism_phi_measure"] == "EMD"
        assert kw["parallel"] is True
        assert kw["precision"] == 6

    def test_as_kwargs_excludes_colliding_formalism_fields(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        kw = snap.as_kwargs()
        for collider in colliding_formalism_fields():
            assert collider not in kw


class TestFieldRouting:
    def test_unique_iit_fields_route_to_iit_subnamespace(self):
        # Pick a non-colliding IIT-only field and verify the routing target.
        assert FIELD_TO_LAYER["mechanism_phi_measure"] == ("formalism", "iit")

    def test_unique_ac_fields_route_to_actual_causation_subnamespace(self):
        # ``measure`` is unique to AC.
        assert FIELD_TO_LAYER["alpha_measure"] == ("formalism", "actual_causation")

    def test_colliding_fields_excluded_from_routing(self):
        # ``mechanism_partition_scheme`` exists in both IIT and AC.
        for name in colliding_formalism_fields():
            assert name not in FIELD_TO_LAYER

    def test_collisions_set_matches_intersection(self):
        iit_names = {f.name for f in fields(IITConfig)}
        ac_names = {f.name for f in fields(ActualCausationConfig)}
        assert colliding_formalism_fields() == iit_names & ac_names

    def test_non_formalism_layer_fields_route_with_none_subnamespace(self):
        for f in fields(NumericsConfig):
            assert FIELD_TO_LAYER[f.name] == ("numerics", None)
        for f in fields(InfrastructureConfig):
            assert FIELD_TO_LAYER[f.name] == ("infrastructure", None)


class TestGlobalConfigFacade:
    def test_layered_reads_work(self):
        assert config.numerics.precision == 13
        assert config.formalism.iit.version == "IIT_4_0_2023"
        assert config.formalism.actual_causation.alpha_measure == "PMI"
        assert config.infrastructure.parallel is False

    def test_legacy_uppercase_read_still_works(self):
        # Uppercase legacy access is preserved for non-colliding leaves.
        assert config.PRECISION == 13
        assert config.PARALLEL is False
        assert config.VERSION == "IIT_4_0_2023"
        assert config.MECHANISM_PHI_MEASURE == "GENERALIZED_INTRINSIC_DIFFERENCE"

    def test_legacy_uppercase_write_propagates_to_layered_view(self):
        original = config.PRECISION
        config.PRECISION = 7
        try:
            assert config.numerics.precision == 7
        finally:
            config.PRECISION = original

    def test_lowercase_layered_write_propagates_to_uppercase_view(self):
        original = config.PRECISION
        config.precision = 9
        try:
            assert config.PRECISION == 9
            assert config.numerics.precision == 9
        finally:
            config.PRECISION = original

    def test_unknown_lowercase_field_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown config option"):
            config.nonexistent_field = 0

    def test_replacing_layer_attribute_works(self):
        original = config.numerics
        try:
            config.numerics = NumericsConfig(precision=6)
            assert config.numerics.precision == 6
            assert config.PRECISION == 6
        finally:
            config.numerics = original

    def test_replacing_layer_with_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="Cannot replace layer"):
            config.numerics = "not a NumericsConfig"  # type: ignore[assignment]

    def test_replacing_iit_subnamespace_works(self):
        original = config.formalism.iit
        try:
            config.iit = IITConfig(mechanism_phi_measure="EMD")
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
            assert config.formalism.actual_causation.alpha_measure == "PMI"
        finally:
            config.iit = original

    def test_replacing_actual_causation_subnamespace_works(self):
        original = config.formalism.actual_causation
        try:
            config.actual_causation = ActualCausationConfig(alpha_measure="KLD")
            assert config.formalism.actual_causation.alpha_measure == "KLD"
        finally:
            config.actual_causation = original

    def test_flat_write_to_unique_iit_field_routes_to_iit(self):
        original = config.formalism.iit.mechanism_phi_measure
        try:
            config.mechanism_phi_measure = "EMD"
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
        finally:
            config.mechanism_phi_measure = original

    def test_flat_write_to_colliding_field_raises(self):
        with pytest.raises(ConfigurationError, match="ambiguous"):
            config.mechanism_partition_scheme = "BI"

    def test_flat_read_of_colliding_field_raises(self):
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = config.mechanism_partition_scheme

    def test_snapshot_returns_config_snapshot(self):
        snap = config.snapshot()
        assert isinstance(snap, ConfigSnapshot)
        assert snap.numerics.precision == config.numerics.precision

    def test_global_config_is_global_config_instance(self):
        assert isinstance(config, _GlobalConfig)
