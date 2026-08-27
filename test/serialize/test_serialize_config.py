"""Round-tripping of the config snapshot attached to result objects."""

import json
from dataclasses import fields

import pytest

import pyphi
from pyphi import examples
from pyphi import serialize
from pyphi.conf import presets
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
from pyphi.models.diff import ResultDiff
from pyphi.models.partitions import NullCut

FORMATS = ["json", "msgpack"]

PRESET_NAMES = ["iit3", "iit4_2023", "iit4_2026"]


def round_trip(obj, fmt="msgpack"):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_sia():
    return SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))


def assert_snapshots_equal_field_for_field(a: ConfigSnapshot, b: ConfigSnapshot):
    for layer_name in ("infrastructure", "numerics"):
        a_layer = getattr(a, layer_name)
        b_layer = getattr(b, layer_name)
        for f in fields(a_layer):
            a_val = getattr(a_layer, f.name)
            b_val = getattr(b_layer, f.name)
            assert a_val == b_val, f"{layer_name}.{f.name}: {a_val!r} != {b_val!r}"
            assert type(a_val) is type(b_val), f"{layer_name}.{f.name} type"
    for sub_name in ("iit", "actual_causation"):
        a_sub = getattr(a.formalism, sub_name)
        b_sub = getattr(b.formalism, sub_name)
        for f in fields(a_sub):
            a_val = getattr(a_sub, f.name)
            b_val = getattr(b_sub, f.name)
            assert a_val == b_val, (
                f"formalism.{sub_name}.{f.name}: {a_val!r} != {b_val!r}"
            )
            assert type(a_val) is type(b_val), f"formalism.{sub_name}.{f.name} type"


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_config_snapshot_round_trips_every_preset(preset_name, fmt):
    with pyphi.config.override(**getattr(presets, preset_name)):
        obj = make_sia()
    assert isinstance(obj.config, ConfigSnapshot)
    restored = round_trip(obj, fmt)
    assert isinstance(restored.config, ConfigSnapshot)
    assert_snapshots_equal_field_for_field(obj.config, restored.config)
    assert restored.config == obj.config


def test_loaded_result_diff_returns_result_diff():
    with pyphi.config.override(**presets.iit4_2023):
        sia = examples.basic_system().sia()
    loaded = round_trip(sia)
    assert isinstance(sia.diff(loaded), ResultDiff)
    assert isinstance(loaded.diff(sia), ResultDiff)
    assert isinstance(loaded.diff(loaded), ResultDiff)


def test_rerun_recipe_works_on_loaded_result():
    with pyphi.config.override(**presets.iit4_2023):
        obj = make_sia()
    loaded = round_trip(obj)
    with pyphi.config.override(**loaded.config.as_overrides()):
        assert pyphi.config.formalism.iit.version == "IIT_4_0_2023"


def test_loaded_iit3_analysis_keeps_big_phi_label():
    # An IIT 3.0 result's Φ label must come from its own stored config,
    # not from the ambient default at load time.
    with pyphi.config.override(**presets.iit3):
        ana = pyphi.analyze(examples.basic_substrate(), (1, 0, 0))
    assert ana._phi_label == "Φ"
    # Load under the ambient (IIT 4.0) default.
    loaded = round_trip(ana)
    assert loaded._phi_label == "Φ"
    assert "Φ" in loaded._describe(0).compact


def test_legacy_frozenmap_repr_config_loads():
    # Payloads written by earlier 2.0 builds stored the parallel-evaluation
    # mappings as FrozenMap repr strings; they must still rehydrate.
    obj = make_sia()
    data = json.loads(serialize.dumps(obj, format="json"))
    infra = data["payload"]["config"]["infrastructure"]
    mangled = [k for k, v in infra.items() if isinstance(v, dict)]
    assert mangled, "expected mapping-valued infrastructure fields"
    for key in mangled:
        infra[key] = f"FrozenMap({infra[key]!r})"
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert isinstance(restored.config, ConfigSnapshot)
    assert_snapshots_equal_field_for_field(obj.config, restored.config)


def test_config_from_unknown_fields_is_ignored():
    # Fields written by other PyPhi versions don't break rehydration.
    obj = make_sia()
    data = json.loads(serialize.dumps(obj, format="json"))
    data["payload"]["config"]["numerics"]["not_a_real_field"] = 42
    data["payload"]["config"]["formalism"]["iit"]["not_a_real_field"] = "x"
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert isinstance(restored.config, ConfigSnapshot)
    assert restored.config == obj.config
