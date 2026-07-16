"""Typed, compact (de)serialization of PyPhi results via msgspec.

Supports two wire formats from one schema: ``"json"`` (default, readable
structure) and ``"msgpack"`` (binary, compact). The document carries a single
top-level ``format_version`` and a node-labels frame written once per
document: nested objects whose labels match the frame store none and inherit
it on decode. A path ending in ``.gz`` is transparently gzip-compressed on
save and decompressed on load.
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
    # A φ value serialized on its own is a native float; every other domain
    # object serializes to a tagged Struct in ``schema.Schema``.
    payload: schema.Schema | float
    # The document's node-labels frame: claimed once by the first labeled
    # payload object, inherited on decode by every object that carries none.
    node_labels: schema.NodeLabelsSchema | None = None


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
    payload, frame = convert.encode_document(obj)
    doc = _Document(format_version=FORMAT_VERSION, payload=payload, node_labels=frame)
    return _encoder(format)(doc)


def loads(data: bytes, *, format: str = "json", node_labels: Any = None) -> Any:
    """Deserialize a document produced by :func:`dumps`.

    Parameters
    ----------
    data : bytes
        The serialized document.
    format : {"json", "msgpack"}, optional
        Wire format. Defaults to ``"json"``.
    node_labels : NodeLabels, optional
        Replacement label frame. If given, it is used in place of the
        document's stored frame; objects carrying their own per-object
        labels keep them.
    """
    doc = _decode(data, format)
    if doc.format_version > FORMAT_VERSION:
        raise ValueError(
            f"cannot load format_version {doc.format_version}: this version of "
            f"PyPhi reads format_version {FORMAT_VERSION} or lower"
        )
    return convert.decode_document(doc.payload, doc.node_labels, node_labels=node_labels)


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
