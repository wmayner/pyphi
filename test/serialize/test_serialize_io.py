import io
import json

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


def test_future_format_version_rejected():
    doc = json.loads(serialize.dumps(1.0, format="json"))
    doc["format_version"] = serialize.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version"):
        serialize.loads(json.dumps(doc).encode(), format="json")


def test_current_format_version_loads():
    data = serialize.dumps(1.0, format="json")
    assert serialize.loads(data, format="json") == 1.0


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_substrate_roundtrip_is_bit_identical(tmp_path, fmt):
    """A substrate with irrational probabilities (sigmoids) round-trips with
    exact array equality and exact dtype in both wire formats."""
    import numpy as np

    from pyphi import config
    from pyphi.substrate import Substrate

    n = 5
    rng = np.random.default_rng(7)
    W = rng.normal(size=(n, n))
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)
    tpm = (1 / (1 + np.exp(-((2 * bits - 1.0) @ W)))).reshape((2,) * n + (n,))
    with config.override(validate_conditional_independence=False):
        sub = Substrate(tpm, cm=np.ones((n, n), dtype=int))

    path = tmp_path / f"sub.{fmt}.gz"
    serialize.save(sub, path)
    loaded = serialize.load(path)
    for factor, back in zip(
        sub.factored_tpm.factors, loaded.factored_tpm.factors, strict=True
    ):
        factor_arr, back_arr = np.asarray(factor), np.asarray(back)
        assert factor_arr.dtype == back_arr.dtype == np.float64
        assert np.array_equal(factor_arr, back_arr)


def _sigmoid_substrate(n, seed=7):
    import numpy as np

    from pyphi import config
    from pyphi.substrate import Substrate

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, n))
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)
    tpm = (1 / (1 + np.exp(-((2 * bits - 1.0) @ W)))).reshape((2,) * n + (n,))
    with config.override(validate_conditional_independence=False):
        return Substrate(tpm, cm=np.ones((n, n), dtype=int))


def test_binary_substrate_serializes_near_raw_size():
    """A binary substrate's serialized size is close to the raw state-by-node
    array — the complementary probability slice of each factor is not stored
    twice."""
    n = 10
    sub = _sigmoid_substrate(n)
    raw = 2**n * n * 8  # float64 state-by-node array
    encoded = serialize.dumps(sub, format="msgpack")
    assert len(encoded) < 1.2 * raw


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_untrimmable_factors_roundtrip_exactly(tmp_path, fmt):
    """Factors whose complementary slice is not the exact float complement
    are stored in full and round-trip bit-identically."""
    import numpy as np

    from pyphi.substrate import Substrate

    rng = np.random.default_rng(0)
    marginals = []
    for _ in range(2):
        f = rng.uniform(size=(2, 2, 2))
        marginals.append(f / f.sum(axis=-1, keepdims=True))
    sub = Substrate(marginals=marginals, state_space=("OFF", "ON"))
    assert any(
        not np.array_equal(np.asarray(f)[..., 0], 1.0 - np.asarray(f)[..., 1])
        for f in sub.factored_tpm.factors
    )

    path = tmp_path / f"sub.{fmt}.gz"
    serialize.save(sub, path)
    loaded = serialize.load(path)
    for factor, back in zip(
        sub.factored_tpm.factors, loaded.factored_tpm.factors, strict=True
    ):
        assert np.array_equal(np.asarray(factor), np.asarray(back))
