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


def test_top_level_save_load(tmp_path, sia):
    import pyphi

    path = tmp_path / "r.json"
    pyphi.save(sia, path)
    assert pyphi.load(path) == sia


@pytest.mark.parametrize(
    "ext,fmt",
    [(".json.gz", "json"), (".msgpack.gz", "msgpack"), (".mpk.gz", "msgpack")],
)
def test_gzip_save_load_roundtrip(tmp_path, sia, ext, fmt):
    import gzip

    path = tmp_path / f"result{ext}"
    serialize.save(sia, path)  # .gz → gzip; wire format from the inner suffix
    assert serialize.load(path) == sia
    # the file really is gzip, and its decompressed bytes are the inner format
    assert path.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic
    assert serialize.loads(gzip.decompress(path.read_bytes()), format=fmt) == sia


def test_gzip_bare_extension_defaults_to_json(tmp_path, sia):
    import gzip

    path = tmp_path / "result.gz"  # no inner wire-format suffix → json
    serialize.save(sia, path)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert serialize.loads(gzip.decompress(path.read_bytes()), format="json") == sia


def test_gzip_with_explicit_format_override(tmp_path, sia):
    import gzip

    path = tmp_path / "result.json.gz"  # .json inner ...
    serialize.save(sia, path, format="msgpack")  # ... overridden to msgpack
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert serialize.loads(gzip.decompress(path.read_bytes()), format="msgpack") == sia
    assert serialize.load(path, format="msgpack") == sia
