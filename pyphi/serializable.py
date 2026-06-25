"""The ``Serializable`` mixin.

Adds ``save``/``load`` convenience to user-facing result types. These methods
are thin delegations to :mod:`pyphi.serialize` — all serialization logic lives
there, never on the domain classes. The ``serialize`` import is deferred to call
time so this module stays free of heavy imports and import cycles.
"""

from __future__ import annotations

from typing import Any


class Serializable:
    """Adds ``save``/``load`` that delegate to :mod:`pyphi.serialize`."""

    def save(self, target: Any, *, format: str | None = None) -> None:
        """Serialize this object to ``target`` (a path or binary file object)."""
        from pyphi import serialize

        serialize.save(self, target, format=format)

    @classmethod
    def load(cls, target: Any, *, format: str | None = None) -> Any:
        """Load an instance of this type from ``target``.

        Raises ``TypeError`` if the file holds a different type.
        """
        from pyphi import serialize

        obj = serialize.load(target, format=format)
        if not isinstance(obj, cls):
            raise TypeError(
                f"{target!r} contains a {type(obj).__name__}, not a {cls.__name__}"
            )
        return obj
