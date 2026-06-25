"""msgspec schema types for serializing PyPhi results.

Each serializable type has one frozen ``msgspec.Struct`` carrying a unique
string ``tag``. ``Schema`` is the tagged union of all of them; msgspec uses the
tag to validate and dispatch on decode. Adding a type means adding its Struct
here and registering its converter in :mod:`pyphi.serialize.convert`.
"""

import msgspec


class DirectionSchema(msgspec.Struct, frozen=True, tag="direction"):
    name: str


# The tagged union grows one member per serializable type.
Schema = DirectionSchema
