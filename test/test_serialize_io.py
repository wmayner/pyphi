import io

import pytest

from pyphi import examples
from pyphi import serialize


@pytest.fixture
def sia():
    return examples.basic_system().sia()


@pytest.mark.parametrize(
    "ext,fmt",
    [(".json", "json"), (".msgpack", "msgpack"), (".mpk", "msgpack")],
)
def test_save_load_roundtrip_by_path(tmp_path, sia, ext, fmt):
    path = tmp_path / f"result{ext}"
    serialize.save(sia, path)  # format inferred from extension
    assert serialize.load(path) == sia
    # the file really is in the inferred format
    assert serialize.loads(path.read_bytes(), format=fmt) == sia


def test_save_load_roundtrip_by_file_object(sia):
    buf = io.BytesIO()
    serialize.save(sia, buf)  # file object defaults to JSON
    buf.seek(0)
    assert serialize.load(buf) == sia


def test_explicit_format_overrides_extension(tmp_path, sia):
    path = tmp_path / "result.json"  # .json suffix ...
    serialize.save(sia, path, format="msgpack")  # ... but written as msgpack
    assert serialize.load(path, format="msgpack") == sia


def test_unknown_extension_defaults_to_json(tmp_path, sia):
    path = tmp_path / "result.dat"
    serialize.save(sia, path)
    assert serialize.loads(path.read_bytes(), format="json") == sia
