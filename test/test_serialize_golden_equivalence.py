"""Equivalence of the new serializer against the trusted jsonify goldens.

The jsonify-format reference fixtures under ``test/data`` are kept as the
trusted reference set: they were produced by the legacy ``pyphi.jsonify``
serializer. This test loads each fixture with ``jsonify`` and asserts that the
new ``pyphi.serialize`` round-trips the resulting domain object to an equal
object, in both JSON and msgpack. It is the strongest validation of the new
serializer because the inputs are real computed results (SIAs, cause-effect
structures with relations, phi-structures), not synthetic instances.

Fixtures that the current code can no longer load (they reference classes
removed in the 2.0 refactor) and bare-container fixtures are skipped with a
recorded reason rather than failing.
"""

from pathlib import Path

import pytest

import pyphi
from pyphi import jsonify
from pyphi import serialize

_DATA_ROOT = Path(__file__).parent / "data"


def _jsonify_fixtures():
    fixtures = []
    for path in sorted(_DATA_ROOT.rglob("*.json")):
        try:
            text = path.read_text()
        except OSError:
            continue
        if jsonify.CLASS_KEY in text:
            fixtures.append(path)
    return fixtures


FIXTURES = _jsonify_fixtures()


def _load_or_skip(path):
    # The trusted reference uses the legacy serializer; version pinning is
    # off for fixtures written by earlier PyPhi releases (as in conftest).
    with pyphi.config.override(validate_json_version=False):
        try:
            with open(path) as f:
                return jsonify.load(f)
        except Exception as exc:
            pytest.skip(f"fixture not loadable by current jsonify: {exc!r}")


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.relative_to(_DATA_ROOT)))
def test_serialize_round_trips_jsonify_golden(path, fmt):
    obj = _load_or_skip(path)
    if isinstance(obj, (list, tuple)):
        pytest.skip("fixture is a bare container, not a single result object")
    restored = serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)
    assert restored == obj
