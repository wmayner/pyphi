"""Typed, compact (de)serialization of PyPhi results via msgspec.

Supports two wire formats from one schema: ``"json"`` (default, readable
structure) and ``"msgpack"`` (binary, compact). The document carries a single
top-level ``format_version``.
"""

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


def dump(obj: Any, fp, *, format: str = "json") -> None:
    fp.write(dumps(obj, format=format))


def load(fp, *, format: str = "json") -> Any:
    return loads(fp.read(), format=format)
