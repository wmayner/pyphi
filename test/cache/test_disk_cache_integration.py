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


def test_config_flip_forces_recompute_not_stale_hit(tmp_path, monkeypatch):
    """A result-affecting config change must miss, not return the previous
    configuration's cached result."""
    _fresh_cache(tmp_path, monkeypatch)
    sub = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(
        **presets.iit4_2023, relation_computation="CONCRETE", disk_cache_results=True
    ):
        first = sub.ces(state)
    with config.override(
        **presets.iit4_2023, relation_computation="ANALYTICAL", disk_cache_results=True
    ):
        second = sub.ces(state)
    assert type(first.relations).__name__ == "ConcreteRelations"
    assert type(second.relations).__name__ == "AnalyticalRelations"


def test_cache_write_failure_does_not_destroy_the_result(monkeypatch):
    """An OSError while persisting must not propagate: the freshly computed
    result (potentially hours of work) is returned; only the cache write is
    lost."""
    from pyphi.cache import disk as disk_module

    def failing_put(self, key, data):
        raise OSError("disk full")

    monkeypatch.setattr(disk_module.DiskCache, "put", failing_put)
    monkeypatch.setattr(disk_module.DiskCache, "get", lambda *_args: None)
    monkeypatch.setattr(disk_module, "result_cache_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr(disk_module.serialize, "dumps", lambda *_a, **_k: b"blob")
    with config.override(disk_cache_results=True):
        result = disk_module.maybe_disk_cached(
            system=None, kind="sia", compute=lambda: "computed", user_kwargs={}
        )
    assert result == "computed"


def test_sia_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    """The key is label-free by design, so a mathematically identical but
    differently-labeled system hits; the result must carry the requester's
    labels, not the computing system's."""
    from pyphi import System
    from pyphi.substrate import Substrate

    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().sia()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).sia()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    assert warm.phi == cold.phi
    assert tuple(warm.node_labels) == ("X", "Y", "Z")
    assert warm.cause is None or tuple(warm.cause.node_labels) == ("X", "Y", "Z")


def test_ces_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    from pyphi import System
    from pyphi.substrate import Substrate

    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        examples.basic_system().ces()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).ces()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    d = next(iter(warm.distinctions))
    assert tuple(d.cause.node_labels) == ("X", "Y", "Z")


def test_iit3_sia_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    from pyphi import System
    from pyphi.substrate import Substrate
    from test.conftest import IIT_3_CONFIG

    _fresh_cache(tmp_path, monkeypatch)
    with IIT_3_CONFIG, config.override(disk_cache_results=True):
        cold = examples.basic_system().sia()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).sia()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    assert warm.phi == cold.phi
    assert tuple(warm.node_labels) == ("X", "Y", "Z")


def test_serialization_failure_does_not_destroy_result(monkeypatch, tmp_path):
    """A cache-write failure of any kind is best-effort: the freshly computed
    result is still returned."""
    from pyphi.cache import disk

    def _fixed_key(*_args, **_kwargs):
        return "test-key"

    def _always_miss(_key):
        return None

    def _unserializable(*_args, **_kwargs):
        raise TypeError("unserializable")

    monkeypatch.setattr(disk, "result_cache_key", _fixed_key, raising=True)
    monkeypatch.setattr(disk._RESULT_DISK_CACHE, "get", _always_miss)
    monkeypatch.setattr(disk.serialize, "dumps", _unserializable)

    class _System:
        node_labels = None

    result = disk.maybe_disk_cached(_System(), "sia", {}, lambda: "computed")
    assert result == "computed"


def test_supplied_system_state_bypasses_the_cache(tmp_path, monkeypatch):
    """A caller-supplied ``system_state`` must not read or write the plain
    ``sia()`` disk entry: nothing verifies it is the canonical state, so
    sharing the entry would let a non-canonical state poison (and be served
    by) the plain result.
    """
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023):
        state = examples.basic_system().sia().system_state  # cache still off
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        # Cold cache: a forced-state call must not populate the plain entry.
        examples.basic_system().sia(system_state=state)
    assert not any(tmp_path.rglob("*")), (
        "system_state calls must not write the disk cache"
    )
