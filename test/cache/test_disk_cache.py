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
