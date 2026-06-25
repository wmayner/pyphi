"""Exact, compact serialization of numpy arrays as ``.npy`` bytes.

The ``.npy`` format records dtype, shape, byte order, and Fortran/C order, so an
array round-trips bit-identically and loads correctly across platforms. Stored in
a ``bytes`` schema field, msgspec emits it as base64 in JSON and as raw bytes in
msgpack. Loaded with ``allow_pickle=False`` so a serialized file can never
execute code on load.
"""

import io

import numpy as np


def array_to_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def bytes_to_array(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)
