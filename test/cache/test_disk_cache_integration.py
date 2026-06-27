"""End-to-end: disk hits equal recomputation; opt-in; bypasses."""

from __future__ import annotations

from pyphi import examples
from pyphi.cache import disk
from pyphi.conf import config
from pyphi.conf import presets


def _fresh_cache(tmp_path, monkeypatch):
    from pyphi import constants

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    disk._RESULT_DISK_CACHE.hits = 0
    disk._RESULT_DISK_CACHE.misses = 0
    # Pin a clean-tree stamp so the key builds regardless of repo state.
    monkeypatch.setattr(disk, "_git_info", lambda: ("testsha", False))


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023):
        examples.basic_system().sia()
    assert not any(tmp_path.rglob("*")), "cache off must create no files"


def test_sia_disk_hit_equals_recompute(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().sia()
        warm = examples.basic_system().sia()  # second call: disk hit
    assert warm == cold
    assert disk._RESULT_DISK_CACHE.hits >= 1


def test_ces_disk_hit_equals_recompute(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().ces()
        warm = examples.basic_system().ces()
    assert warm == cold


def test_kwargs_bypass_the_cache(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    from pyphi.measures.distribution import resolve_system_measure

    with config.override(**presets.iit4_2023, disk_cache_results=True):
        # passing an explicit measure kwarg must bypass (key can't capture it)
        examples.basic_system().sia(
            system_measure=resolve_system_measure(
                config.formalism.iit.system_phi_measure
            )
        )
    assert not any(tmp_path.rglob("*")), "explicit kwargs must not be cached"
