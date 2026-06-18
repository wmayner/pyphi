from __future__ import annotations

import importlib.metadata
from unittest import mock

from pyphi.provenance import Provenance


def test_capture_populates_fields():
    prov = Provenance.capture()
    assert prov.pyphi_version == importlib.metadata.version("pyphi")
    assert isinstance(prov.timestamp, str) and prov.timestamp.endswith("+00:00")
    assert prov.python_version.count(".") == 2
    assert isinstance(prov.numpy_version, str) and prov.numpy_version
    assert isinstance(prov.scipy_version, str) and prov.scipy_version
    assert "/" in prov.platform
    assert prov.wall_time is None
    assert prov.seed is None
    # git fields are either both populated or both None
    assert (prov.git_sha is None) == (prov.git_dirty is None)


def test_capture_passes_through_wall_time_and_seed():
    prov = Provenance.capture(wall_time=1.5, seed=42)
    assert prov.wall_time == 1.5
    assert prov.seed == 42


def test_git_info_fallback_when_not_a_repo():
    from pyphi import provenance

    provenance._git_info.cache_clear()
    with mock.patch(
        "pyphi.provenance.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    ):
        sha, dirty = provenance._git_info()
    assert sha is None
    assert dirty is None
    provenance._git_info.cache_clear()


def test_jsonify_round_trip():
    from pyphi.jsonify import dumps
    from pyphi.jsonify import loads

    prov = Provenance.capture(wall_time=2.0, seed=7)
    restored = loads(dumps(prov))
    assert isinstance(restored, Provenance)
    assert restored == prov
