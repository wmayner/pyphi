"""Typed, compact (de)serialization of PyPhi results via msgspec.

Supports two wire formats from one schema: ``"json"`` (default, readable
structure) and ``"msgpack"`` (binary, compact). The document carries a single
top-level ``format_version``.
"""

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


def _infer_format(target: Any, format: str | None) -> str:
    if format is not None:
        return format
    if isinstance(target, (str, os.PathLike)):
        return _SUFFIX_FORMATS.get(Path(target).suffix.lower(), "json")
    return "json"


def save(obj: Any, target: Any, *, format: str | None = None) -> None:
    """Serialize ``obj`` to ``target`` (a path or an open binary file object).

    The wire format is inferred from a path's extension (``.json`` →
    ``"json"``; ``.msgpack`` / ``.mpk`` → ``"msgpack"``; otherwise ``"json"``)
    unless ``format`` is given.
    """
    data = dumps(obj, format=_infer_format(target, format))
    if isinstance(target, (str, os.PathLike)):
        with open(target, "wb") as f:
            f.write(data)
    else:
        target.write(data)


def load(target: Any, *, format: str | None = None) -> Any:
    """Deserialize from ``target`` (a path or an open binary file object)."""
    fmt = _infer_format(target, format)
    if isinstance(target, (str, os.PathLike)):
        with open(target, "rb") as f:
            data = f.read()
    else:
        data = target.read()
    return loads(data, format=fmt)
