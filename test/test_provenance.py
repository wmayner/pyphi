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


def _basic_iit4_sia():
    """An IIT 4.0 SIA computed through the public dispatch entry point."""
    from pyphi import examples

    return examples.basic_system().sia()


def test_iit4_sia_carries_provenance():
    sia = _basic_iit4_sia()
    assert isinstance(sia.provenance, Provenance)


def test_provenance_does_not_pollute_equality():
    # Two independent runs differ in timestamp but must stay equal.
    a = _basic_iit4_sia()
    b = _basic_iit4_sia()
    assert a == b


def test_provenance_excluded_from_diff():
    a = _basic_iit4_sia()
    b = _basic_iit4_sia()
    d = a.diff(b)
    assert float(d.delta_phi) == 0
    assert d.config_diff == {}


def test_every_config_carrying_result_carries_provenance():
    import pyphi
    from pyphi import actual
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.conf.snapshot import ConfigSnapshot
    from pyphi.direction import Direction
    from pyphi.formalism import iit3

    system = examples.basic_system()
    results = [system.sia(), system.ces()]  # IIT 4.0 SIA + CES
    with pyphi.config.override(**presets.iit3):
        results.append(iit3.sia(system))  # IIT 3.0 SIA
    transition = examples.prevention_transition()
    results.append(actual.sia(transition, Direction.BIDIRECTIONAL))  # AcSIA

    for result in results:
        assert isinstance(result.config, ConfigSnapshot)
        assert isinstance(result.provenance, Provenance)


def test_entry_point_sets_wall_time():
    sia = _basic_iit4_sia()
    assert sia.provenance.wall_time is not None
    assert sia.provenance.wall_time >= 0.0


def test_ces_entry_point_sets_wall_time():
    from pyphi import examples

    ces = examples.basic_system().ces()
    assert ces.provenance.wall_time is not None
    assert ces.provenance.wall_time >= 0.0


def test_direct_construction_has_no_wall_time():
    # A result built without going through the entry point keeps wall_time=None.
    from pyphi.provenance import Provenance

    assert Provenance.capture().wall_time is None
