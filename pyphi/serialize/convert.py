"""Convert between PyPhi domain objects and their msgspec schema Structs.

Two registries map a domain type to its encoder and a schema Struct type to its
decoder. This replaces the per-class ``to_json`` / ``from_json`` methods. Each
serializable type adds one ``_register_<type>()`` populating both registries,
all called at import time.
"""

from collections.abc import Callable
from typing import Any

from pyphi.direction import Direction

from . import schema

_ENCODERS: dict[type, Callable[[Any], Any]] = {}  # domain type   -> encode
_DECODERS: dict[type, Callable[[Any], Any]] = {}  # schema Struct  -> decode


def to_schema(obj: Any) -> Any:
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
    decode = _DECODERS.get(type(struct))
    if decode is None:
        raise TypeError(f"No deserializer registered for {type(struct).__name__}")
    return decode(struct)


def _register_direction() -> None:
    _ENCODERS[Direction] = lambda d: schema.DirectionSchema(name=d.name)
    _DECODERS[schema.DirectionSchema] = lambda s: Direction[s.name]


_register_direction()
