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


def test_serialize_round_trip():
    from pyphi import serialize

    prov = Provenance.capture(wall_time=2.0, seed=7)
    restored = serialize.loads(serialize.dumps(prov))
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


def test_provenance_shown_only_at_level_4():
    from pyphi import config

    sia = _basic_iit4_sia()
    with config.override(repr_verbosity=3):
        assert "Provenance" not in repr(sia)
    with config.override(repr_verbosity=4):
        text = repr(sia)
        assert "Provenance" in text
        assert sia.provenance.pyphi_version in text


def test_repr_verbosity_4_is_valid_and_5_is_rejected():
    import pytest

    from pyphi import config

    with config.override(repr_verbosity=4):
        pass  # must not raise
    with pytest.raises(ValueError), config.override(repr_verbosity=5):
        pass


def test_with_provenance_updates_fields_and_returns_self_mutable():
    # IIT 4.0 SIA is a (non-frozen) dataclass: update in place.
    sia = _basic_iit4_sia()
    returned = sia.with_provenance(note="experiment 1", seed=42)
    assert returned is sia
    assert sia.provenance.note == "experiment 1"
    assert sia.provenance.seed == 42


def test_with_provenance_on_frozen_result():
    # CauseEffectStructure is a frozen dataclass: the frozen-replace path.
    from pyphi import examples

    ces = examples.basic_system().ces()
    ces.with_provenance(note="frozen ok")
    assert ces.provenance.note == "frozen ok"


def test_with_provenance_preserves_other_fields():
    sia = _basic_iit4_sia()
    version = sia.provenance.pyphi_version
    wall = sia.provenance.wall_time
    sia.with_provenance(note="keep the rest")
    assert sia.provenance.pyphi_version == version
    assert sia.provenance.wall_time == wall


def test_with_provenance_rejects_unknown_field():
    import pytest

    sia = _basic_iit4_sia()
    with pytest.raises(TypeError):
        sia.with_provenance(not_a_field=1)


def test_note_renders_at_level_4_when_set():
    from pyphi import config

    sia = _basic_iit4_sia()
    sia.with_provenance(note="visible note")
    with config.override(repr_verbosity=4):
        assert "visible note" in repr(sia)


def test_note_round_trips_through_serialize():
    from pyphi import serialize

    prov = Provenance.capture(seed=1).with_wall_time(0.5)
    from dataclasses import replace

    prov = replace(prov, note="round-trip")
    restored = serialize.loads(serialize.dumps(prov))
    assert restored.note == "round-trip"
    assert restored == prov
