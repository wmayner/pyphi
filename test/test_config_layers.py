"""Tests for the layered config dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace

import pytest

from pyphi.conf import config
from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.legacy_global import _GlobalConfig
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


class TestFormalismConfig:
    def test_defaults_match_legacy(self):
        cfg = FormalismConfig()
        assert cfg.formalism == "IIT_4_0_2023"
        assert cfg.repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert cfg.partition_type == "ALL"
        assert cfg.system_partition_type == "SET_UNI/BI"
        assert cfg.shortcircuit_sia is True

    def test_is_frozen(self):
        cfg = FormalismConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.formalism = "IIT_3_0"  # type: ignore[misc]

    def test_replace_returns_new_instance(self):
        a = FormalismConfig()
        b = replace(a, repertoire_distance="EMD")
        assert a.repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert b.repertoire_distance == "EMD"

    def test_mip_tie_resolution_default_is_list(self):
        cfg = FormalismConfig()
        assert cfg.mip_tie_resolution == ["NORMALIZED_PHI", "NEGATIVE_PHI"]

    def test_mip_tie_resolution_each_instance_independent(self):
        a = FormalismConfig()
        b = FormalismConfig()
        assert a.mip_tie_resolution is not b.mip_tie_resolution


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
        assert snap.formalism.formalism == "IIT_4_0_2023"

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
            formalism=FormalismConfig(repertoire_distance="EMD"),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(precision=6),
        )
        diff = a.diff(b)
        assert diff == {
            "formalism.repertoire_distance": (
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
            formalism=FormalismConfig(repertoire_distance="EMD"),
            infrastructure=InfrastructureConfig(parallel=True),
            numerics=NumericsConfig(precision=6),
        )
        kw = snap.as_kwargs()
        assert kw["repertoire_distance"] == "EMD"
        assert kw["parallel"] is True
        assert kw["precision"] == 6


class TestFieldRouting:
    def test_all_layer_fields_present(self):
        from dataclasses import fields

        for layer_name, layer_cls in [
            ("formalism", FormalismConfig),
            ("infrastructure", InfrastructureConfig),
            ("numerics", NumericsConfig),
        ]:
            for f in fields(layer_cls):
                assert FIELD_TO_LAYER[f.name] == layer_name

    def test_no_collisions_in_current_layers(self):
        from dataclasses import fields

        all_fields: list[str] = []
        for layer_cls in (FormalismConfig, InfrastructureConfig, NumericsConfig):
            all_fields.extend(f.name for f in fields(layer_cls))
        assert len(all_fields) == len(set(all_fields)), (
            f"Collision in field names: {sorted(all_fields)}"
        )

    def test_collision_raises(self):
        from dataclasses import dataclass

        from pyphi.conf._field_routing import _build_field_map

        @dataclass(frozen=True)
        class _LayerA:
            x: int = 0

        @dataclass(frozen=True)
        class _LayerB:
            x: int = 0

        with pytest.raises(ConfigurationError, match="Config field name collision"):
            _build_field_map([("a", _LayerA), ("b", _LayerB)])


class TestGlobalConfigFacade:
    def test_layered_reads_work(self):
        assert config.numerics.precision == 13
        assert config.formalism.formalism == "IIT_4_0_2023"
        assert config.infrastructure.parallel is False

    def test_legacy_uppercase_read_still_works(self):
        # During the cutover, both access patterns reflect the same source
        # of truth (the wrapped legacy PyphiConfig instance).
        assert config.PRECISION == 13
        assert config.FORMALISM == "IIT_4_0_2023"
        assert config.PARALLEL is False

    def test_legacy_uppercase_write_propagates_to_layered_view(self):
        original = config.PRECISION
        config.PRECISION = 7
        try:
            assert config.numerics.precision == 7
        finally:
            config.PRECISION = original

    def test_lowercase_layered_write_propagates_to_legacy(self):
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

    def test_replacing_layer_attribute_raises(self):
        # Wholesale layer replacement is intentionally blocked during the
        # cutover; layers are computed views, not stored state.
        with pytest.raises(ConfigurationError, match="Cannot replace layer"):
            config.numerics = NumericsConfig(precision=6)

    def test_snapshot_returns_config_snapshot(self):
        snap = config.snapshot()
        assert isinstance(snap, ConfigSnapshot)
        assert snap.numerics.precision == config.numerics.precision

    def test_global_config_is_global_config_instance(self):
        assert isinstance(config, _GlobalConfig)
