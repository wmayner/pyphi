"""Typed, compact (de)serialization of PyPhi results via msgspec.

Supports two wire formats from one schema: ``"json"`` (default, readable
structure) and ``"msgpack"`` (binary, compact). The document carries a single
top-level ``format_version``. A path ending in ``.gz`` is transparently
gzip-compressed on save and decompressed on load.
"""

import gzip
import os
from pathlib import Path
from typing import Any

import msgspec

from . import convert
from . import schema

FORMAT_VERSION = 1


class _Document(msgspec.Struct, frozen=True):
    format_version: int
    payload: schema.Schema


def _encoder(fmt: str):
    if fmt == "json":
        return msgspec.json.encode
    if fmt == "msgpack":
        return msgspec.msgpack.encode
    raise ValueError(f"Unknown format: {fmt!r}")


def _decode(data: bytes, fmt: str) -> _Document:
    if fmt == "json":
        return msgspec.json.decode(data, type=_Document)
    if fmt == "msgpack":
        return msgspec.msgpack.decode(data, type=_Document)
    raise ValueError(f"Unknown format: {fmt!r}")


def dumps(obj: Any, *, format: str = "json") -> bytes:
    doc = _Document(format_version=FORMAT_VERSION, payload=convert.to_schema(obj))
    return _encoder(format)(doc)


def loads(data: bytes, *, format: str = "json") -> Any:
    doc = _decode(data, format)
    return convert.from_schema(doc.payload)


_SUFFIX_FORMATS = {".json": "json", ".msgpack": "msgpack", ".mpk": "msgpack"}


def _is_gzip_path(target: Any) -> bool:
    return isinstance(target, (str, os.PathLike)) and str(target).lower().endswith(".gz")


def _infer_format(target: Any, format: str | None) -> str:
    if format is not None:
        return format
    if isinstance(target, (str, os.PathLike)):
        suffixes = [s.lower() for s in Path(target).suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]  # a .gz wraps the inner wire-format suffix
        suffix = suffixes[-1] if suffixes else ""
        return _SUFFIX_FORMATS.get(suffix, "json")
    return "json"


def save(obj: Any, target: Any, *, format: str | None = None) -> None:
    """Serialize ``obj`` to ``target``.

    Parameters
    ----------
    obj : Any
        A PyPhi domain object with a registered serializer (see
        :mod:`pyphi.serialize.convert` for the supported types).
    target : str or os.PathLike or file object
        Destination path, or an open binary file object to write to.
    format : {"json", "msgpack"}, optional
        Wire format. If None (the default), it is inferred from a path's
        extension: ``.json`` gives ``"json"``; ``.msgpack`` or ``.mpk`` give
        ``"msgpack"``; any other extension, or a non-path target, gives
        ``"json"``.

    Notes
    -----
    A path ending in ``.gz`` is gzip-compressed, with the wire format taken
    from the inner suffix (``result.json.gz`` yields gzip-compressed JSON).
    Compression is applied only to path targets, not to file objects.
    """
    data = dumps(obj, format=_infer_format(target, format))
    if isinstance(target, (str, os.PathLike)):
        opener = gzip.open if _is_gzip_path(target) else open
        with opener(target, "wb") as f:
            f.write(data)
    else:
        target.write(data)


def load(target: Any, *, format: str | None = None) -> Any:
    """Deserialize a PyPhi domain object from ``target``.

    Parameters
    ----------
    target : str or os.PathLike or file object
        Source path, or an open binary file object to read from.
    format : {"json", "msgpack"}, optional
        Wire format. If None (the default), it is inferred from a path's
        extension exactly as in :func:`save`.

    Returns
    -------
    Any
        The reconstructed PyPhi domain object.

    Notes
    -----
    A path ending in ``.gz`` is transparently decompressed.
    """
    fmt = _infer_format(target, format)
    if isinstance(target, (str, os.PathLike)):
        opener = gzip.open if _is_gzip_path(target) else open
        with opener(target, "rb") as f:
            data = f.read()
    else:
        data = target.read()
    return loads(data, format=fmt)
