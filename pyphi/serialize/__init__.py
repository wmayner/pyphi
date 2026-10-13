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
import zlib
from pathlib import Path
from typing import Any

import msgspec

from . import convert
from . import schema

FORMAT_VERSION = 2


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


_DECODERS = {"json": msgspec.json.decode, "msgpack": msgspec.msgpack.decode}


def _decode(data: bytes, fmt: str) -> _Document:
    if fmt not in _DECODERS:
        raise ValueError(f"Unknown format: {fmt!r}")
    try:
        return _DECODERS[fmt](data, type=_Document)
    except msgspec.MsgspecError as error:
        raise ValueError(_unreadable(data, fmt, error)) from error


def _unreadable(data: bytes, fmt: str, error: Exception) -> str:
    """Say why ``data`` is not a document this version of PyPhi can read."""
    try:
        content = _DECODERS[fmt](data)
    except msgspec.MsgspecError:
        hint = (
            " If it was saved as msgpack, pass format='msgpack'."
            if fmt == "json"
            else ""
        )
        return f"the data is not valid {fmt} ({error}).{hint}"
    if isinstance(content, dict) and {"__class__", "__version__"} & content.keys():
        return (
            "this file was written by the jsonify module of PyPhi 1.x "
            f"(version {content.get('__version__', 'unknown')}), which PyPhi 2 "
            "cannot read; recompute the result. A saved 1.x network can be "
            "rebuilt from its JSON as pyphi.Substrate(data['tpm'], cm=data['cm'])."
        )
    if not isinstance(content, dict) or "format_version" not in content:
        return "not a file saved by pyphi.save: it has no format_version."
    return (
        f"the file does not match this version's format ({error}); it may have "
        "been saved by a newer version of PyPhi."
    )


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

    Raises
    ------
    ValueError
        If ``data`` is not a document produced by :func:`dumps` in a format
        this version reads; the message says what the data appears to be.
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

    Raises
    ------
    ValueError
        If the data is not a document saved by :func:`save`, for example a
        PyPhi 1.x ``jsonify`` file. The error's notes give the path.

    Notes
    -----
    A path ending in ``.gz`` is transparently decompressed.
    """
    fmt = _infer_format(target, format)
    try:
        if isinstance(target, (str, os.PathLike)):
            if _is_gzip_path(target):
                try:
                    with gzip.open(target, "rb") as f:
                        data = f.read()
                except (EOFError, gzip.BadGzipFile, zlib.error) as error:
                    raise ValueError(
                        "the file is not a complete gzip stream; it may be "
                        f"truncated or not compressed ({error})"
                    ) from error
            else:
                with open(target, "rb") as f:
                    data = f.read()
        else:
            data = target.read()
        return loads(data, format=fmt)
    except ValueError as error:
        if isinstance(target, (str, os.PathLike)):
            error.add_note(f"while loading {os.fspath(target)}")
        raise
