"""Disk-backed result store: round-trip, atomic writes, corruption tolerance."""

from __future__ import annotations

from pyphi.cache.disk import DiskCache
from pyphi.cache.disk import _decode_or_none


def test_put_get_round_trip(tmp_path, monkeypatch):
    from pyphi import constants

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    cache = DiskCache("test.disk", "results_t1")
    assert cache.get("abc") is None  # miss before put
    cache.put("abc", b"hello")
    assert cache.get("abc") == b"hello"
    assert cache.size == 1
    cache.clear()
    assert cache.get("abc") is None
    assert cache.size == 0


def test_decode_or_none_tolerates_corruption():
    assert _decode_or_none(b"not a valid record") is None


def test_puts_use_distinct_temp_files(tmp_path, monkeypatch):
    """Guards defect: the temp name was keyed on pid alone, so two threads
    writing the same key collided on the temp path and one raised
    ``FileNotFoundError``."""
    from pathlib import Path

    from pyphi import constants
    from pyphi.cache import registry as reg

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    cache = DiskCache("test.disk.tmpname", "results_tmpname")
    recorded = []
    original = Path.write_bytes

    def recording(self, data):
        recorded.append(str(self))
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", recording)
    cache.put("k", b"one")
    cache.put("k", b"two")
    reg.unregister("test.disk.tmpname")
    assert len(recorded) == 2
    assert recorded[0] != recorded[1]
    assert cache.get("k") == b"two"


def test_clear_all_spares_the_persistent_disk_store(tmp_path, monkeypatch):
    """Guards defect: registry-wide ``clear_all()`` — whose purpose is
    recovering memory — silently deleted the durable on-disk result cache.
    An explicit ``clear(name)`` still clears it."""
    from pyphi import constants
    from pyphi.cache import registry as reg

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    cache = DiskCache("test.disk.persistent", "results_persist")
    cache.put("abc", b"durable")
    try:
        reg.clear_all()
        assert cache.get("abc") == b"durable"
        reg.clear("test.disk.persistent")
        assert cache.get("abc") is None
    finally:
        reg.unregister("test.disk.persistent")
