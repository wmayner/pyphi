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
        assert cfg.mechanism_partition_scheme == "JOINT_PARTITION_ALL"
        assert cfg.system_partition_scheme == "DIRECTED_SET_PARTITION"
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
        assert cfg.mechanism_partition_scheme == "JOINT_PARTITION_ALL"
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
        assert config.CES_MEASURE == "SUM_SMALL_PHI"
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
            config.mechanism_partition_scheme = "JOINT_BIPARTITION"

    def test_flat_read_of_colliding_field_raises(self):
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = config.mechanism_partition_scheme

    def test_snapshot_returns_config_snapshot(self):
        snap = config.snapshot()
        assert isinstance(snap, ConfigSnapshot)
        assert snap.numerics.precision == config.numerics.precision

    def test_global_config_is_global_config_instance(self):
        assert isinstance(config, _GlobalConfig)


class TestDottedPathAccessor:
    def test_read_top_level_layer_field(self):
        assert config["numerics.precision"] == config.numerics.precision

    def test_read_nested_formalism_field(self):
        assert (
            config["formalism.iit.mechanism_phi_measure"]
            == config.formalism.iit.mechanism_phi_measure
        )

    def test_read_actual_causation_field(self):
        assert (
            config["formalism.actual_causation.alpha_measure"]
            == config.formalism.actual_causation.alpha_measure
        )

    def test_read_infrastructure_field(self):
        assert config["infrastructure.parallel"] == config.infrastructure.parallel

    def test_write_top_level_layer_field(self):
        original = config.numerics.precision
        try:
            config["numerics.precision"] = 7
            assert config.numerics.precision == 7
            assert config["numerics.precision"] == 7
        finally:
            config["numerics.precision"] = original

    def test_write_nested_formalism_field(self):
        original = config.formalism.iit.mechanism_phi_measure
        try:
            config["formalism.iit.mechanism_phi_measure"] = "EMD"
            assert config.formalism.iit.mechanism_phi_measure == "EMD"
        finally:
            config["formalism.iit.mechanism_phi_measure"] = original

    def test_write_actual_causation_field(self):
        original = config.formalism.actual_causation.alpha_measure
        try:
            config["formalism.actual_causation.alpha_measure"] = "KLD"
            assert config.formalism.actual_causation.alpha_measure == "KLD"
        finally:
            config["formalism.actual_causation.alpha_measure"] = original

    def test_read_unknown_path_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown config path"):
            _ = config["formalism.iit.nonexistent_field"]

    def test_read_empty_path_raises_keyerror(self):
        with pytest.raises(KeyError, match="Invalid config path"):
            _ = config[""]

    def test_write_empty_path_raises_keyerror(self):
        with pytest.raises(KeyError, match="Path must address a leaf"):
            config[""] = 1

    def test_write_single_part_path_raises_keyerror(self):
        with pytest.raises(KeyError, match="Path must address a leaf"):
            config["precision"] = 1

    def test_write_unknown_layer_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown top-level layer"):
            config["bogus.precision"] = 1

    def test_write_unknown_field_within_known_layer_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown config path"):
            config["numerics.nonexistent_field"] = 1


# ---------------------------------------------------------------------------
# Mapping protocol surface
# ---------------------------------------------------------------------------


class TestConfigMappingProtocol:
    """Pin the Mapping protocol behavior on the global config facade."""

    def test_iter_yields_all_leaves(self):
        """iter(config) yields every leaf as a dotted path in declaration order."""
        keys = list(config)
        assert "numerics.precision" in keys
        assert "formalism.iit.version" in keys
        assert "formalism.iit.mechanism_phi_measure" in keys
        assert "formalism.actual_causation.alpha_measure" in keys
        assert "infrastructure.parallel" in keys
        assert "numerics" not in keys
        assert "formalism" not in keys
        assert "formalism.iit" not in keys

    def test_len_matches_iter(self):
        """len(config) equals the number of paths yielded by iteration."""
        assert len(config) == len(list(iter(config)))
        assert len(config) > 0

    def test_contains_dotted_path(self):
        """'numerics.precision' in config is True."""
        assert "numerics.precision" in config
        assert "formalism.iit.mechanism_phi_measure" in config
        assert "infrastructure.parallel" in config

    def test_contains_bare_leaf(self):
        """A bare leaf name resolves via FIELD_TO_LAYER routing."""
        assert "precision" in config
        assert "mechanism_phi_measure" in config

    def test_contains_invalid_path(self):
        """An unknown path is not contained."""
        assert "foo.bar" not in config
        assert "numerics.nonexistent" not in config
        assert "nonexistent" not in config

    def test_contains_non_string(self):
        """Non-string keys are not contained (and do not raise)."""
        assert 42 not in config
        assert None not in config
        assert ("numerics", "precision") not in config

    def test_get_existing_dotted_path(self):
        """config.get('numerics.precision') matches attribute access."""
        assert config.get("numerics.precision") == config.numerics.precision

    def test_get_existing_bare_leaf(self):
        """config.get('precision') routes via FIELD_TO_LAYER."""
        assert config.get("precision") == config.numerics.precision

    def test_get_missing_returns_default(self):
        """config.get on a missing path returns the default."""
        assert config.get("nonexistent", 42) == 42
        assert config.get("foo.bar.baz") is None

    def test_keys_values_items_consistent(self):
        """keys, values, and items agree on order and content."""
        keys = config.keys()
        values = config.values()
        items = config.items()
        assert len(keys) == len(values) == len(items)
        for i, key in enumerate(keys):
            assert items[i] == (key, values[i])
            assert config[key] == values[i]

    def test_items_round_trip(self):
        """Capturing items, mutating, and restoring via __setitem__ recovers state."""
        original_precision = config.numerics.precision
        try:
            config["numerics.precision"] = 99
            assert config["numerics.precision"] == 99
            config["numerics.precision"] = original_precision
            assert config.numerics.precision == original_precision
        finally:
            config["numerics.precision"] = original_precision

    def test_getitem_bare_leaf(self):
        """config['precision'] returns config.numerics.precision."""
        assert config["precision"] == config.numerics.precision
        assert (
            config["mechanism_phi_measure"] == config.formalism.iit.mechanism_phi_measure
        )

    def test_getitem_unknown_bare_leaf(self):
        """An unknown bare leaf key raises KeyError."""
        with pytest.raises(KeyError, match="Unknown config path"):
            config["nonexistent"]


def test_dotted_subscript_subnamespace_shorthand_read():
    assert config["iit.version"] == config["formalism.iit.version"]
    assert (
        config["actual_causation.alpha_measure"]
        == config["formalism.actual_causation.alpha_measure"]
    )


def test_dotted_subscript_subnamespace_shorthand_write():
    original = config["formalism.iit.mechanism_phi_measure"]
    try:
        config["iit.mechanism_phi_measure"] = "EMD"
        assert config["formalism.iit.mechanism_phi_measure"] == "EMD"
    finally:
        config["formalism.iit.mechanism_phi_measure"] = original


def test_override_dotted_positional_dict():
    before = config["formalism.iit.version"]
    with config.override({"iit.version": "IIT_3_0"}):
        assert config["formalism.iit.version"] == "IIT_3_0"
    assert config["formalism.iit.version"] == before


def test_override_dotted_via_kwargs():
    before = config["formalism.iit.version"]
    with config.override(**{"iit.version": "IIT_3_0"}):
        assert config["formalism.iit.version"] == "IIT_3_0"
    assert config["formalism.iit.version"] == before


def test_override_mixed_dotted_and_flat():
    before_v = config["formalism.iit.version"]
    before_p = config["numerics.precision"]
    with config.override({"iit.version": "IIT_3_0"}, precision=6):
        assert config["formalism.iit.version"] == "IIT_3_0"
        assert config["numerics.precision"] == 6
    assert config["formalism.iit.version"] == before_v
    assert config["numerics.precision"] == before_p


def test_colliding_setattr_error_mentions_dotted_form():
    with pytest.raises(ConfigurationError) as exc:
        config.mechanism_partition_scheme = "JOINT_BIPARTITION"
    msg = str(exc.value)
    assert "override(" in msg
    assert "iit.mechanism_partition_scheme" in msg


def test_colliding_getattr_error_mentions_dotted_form():
    with pytest.raises(AttributeError) as exc:
        _ = config.mechanism_partition_scheme
    assert 'config["iit.mechanism_partition_scheme"]' in str(exc.value)


def test_distinction_phi_normalization_warning_states_transition():
    """The cache-staleness warning names the value change, so the enter and
    exit transitions of an override are distinguishable."""
    import warnings

    import pyphi
    from pyphi.warnings import PyPhiWarning

    old = pyphi.config.formalism.iit.distinction_phi_normalization
    assert old == "NUM_CONNECTIONS_CUT"
    new = "NONE"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pyphi.config.override(**{"iit.distinction_phi_normalization": new}):
            pass
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, PyPhiWarning)
        and "distinction_phi_normalization" in str(w.message)
    ]
    assert len(messages) == 2
    assert f"{old!r} -> {new!r}" in messages[0]
    assert f"{new!r} -> {old!r}" in messages[1]
