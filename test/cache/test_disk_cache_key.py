"""Cache-key builder: separates what changes a result, reuses what doesn't."""

from __future__ import annotations

import pytest

from pyphi import examples
from pyphi.cache import disk
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.substrate import Substrate


@pytest.fixture(autouse=True)
def _clean_tree(monkeypatch):
    """Force a clean-tree git stamp so keys are built regardless of repo state.

    The real working tree is dirty during development, which would make every
    key ``None``; pin a deterministic clean stamp here. The dirty-tree test
    overrides this.
    """
    monkeypatch.setattr(disk, "_git_info", lambda: ("testsha", False))


def test_key_is_hex_str_and_deterministic():
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        k1 = disk.result_cache_key(s, "sia", config.snapshot())
        k2 = disk.result_cache_key(s, "sia", config.snapshot())
        assert isinstance(k1, str) and k1 == k2
        int(k1, 16)  # hex


def test_kind_separates_sia_from_ces():
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        k_sia = disk.result_cache_key(s, "sia", config.snapshot())
        k_ces = disk.result_cache_key(s, "ces", config.snapshot())
        assert k_sia != k_ces


def test_config_separates_formalism_versions():
    s = examples.basic_system()
    with config.override(**presets.iit4_2023):
        snap_2023 = config.snapshot()
    with config.override(**presets.iit4_2026):
        snap_2026 = config.snapshot()
    k_2023 = disk.result_cache_key(s, "sia", snap_2023)
    k_2026 = disk.result_cache_key(s, "sia", snap_2026)
    assert k_2023 != k_2026


def test_relabeled_equivalent_system_shares_key():
    from pyphi import System

    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        relabeled = Substrate.from_factored(
            s.substrate.factored_tpm,
            cm=s.substrate.cm,
            node_labels=("X", "Y", "Z"),
        )
        s2 = System(relabeled, s.state)
        k1 = disk.result_cache_key(s, "sia", config.snapshot())
        k2 = disk.result_cache_key(s2, "sia", config.snapshot())
        assert k1 == k2


def test_dirty_tree_returns_none(monkeypatch):
    monkeypatch.setattr(disk, "_git_info", lambda: ("abc123", True))
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        assert disk.result_cache_key(s, "sia", config.snapshot()) is None


def test_config_digest_covers_every_formalism_and_numerics_field():
    """Complete by construction: no result-affecting config field may be
    missing from the key digest (an omitted field means a config change can
    silently return the previous configuration's cached result)."""
    import dataclasses

    from pyphi.conf.formalism import ActualCausationConfig
    from pyphi.conf.formalism import IITConfig
    from pyphi.conf.numerics import NumericsConfig

    digest = disk._config_digest(config.snapshot()).decode()
    for cls in (IITConfig, ActualCausationConfig, NumericsConfig):
        for f in dataclasses.fields(cls):
            assert f.name in digest, f"digest omits {cls.__name__}.{f.name}"


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("system_partition_include_total", True),
        ("relation_computation", "CONCRETE"),
        ("assume_partitions_cannot_create_new_concepts", True),
        ("shortcircuit_sia", False),
        ("shortcircuit_distinctions", False),
        ("single_micro_nodes_with_selfloops_have_phi", False),
        ("precision", 6),
    ],
)
def test_config_separates_result_affecting_fields(field, alternate):
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        base = disk.result_cache_key(s, "ces", config.snapshot())
    with config.override(**presets.iit4_2023, **{field: alternate}):
        changed = disk.result_cache_key(s, "ces", config.snapshot())
    assert base != changed, f"key ignores result-affecting field {field!r}"
