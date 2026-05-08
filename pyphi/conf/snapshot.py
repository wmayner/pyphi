"""Frozen snapshot of the three config layers, attached to result objects.

A :class:`ConfigSnapshot` mirrors the live ``pyphi.config`` shape but is
immutable: once a result object carries a snapshot, mutating the live
global doesn't change the snapshot's view of what produced the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable snapshot of the three config layers at construction time.

    Result objects carry one of these so reproducibility is self-contained:
    rerunning a saved result is
    ``pyphi.config.override(**snap.as_kwargs())``.
    """

    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def diff(self, other: ConfigSnapshot) -> dict[str, tuple[Any, Any]]:
        """Return ``{dotted-path: (self_value, other_value)}`` for fields that differ."""
        result: dict[str, tuple[Any, Any]] = {}
        for layer_name in ("formalism", "infrastructure", "numerics"):
            self_layer = getattr(self, layer_name)
            other_layer = getattr(other, layer_name)
            for f in fields(self_layer):
                self_val = getattr(self_layer, f.name)
                other_val = getattr(other_layer, f.name)
                if self_val != other_val:
                    result[f"{layer_name}.{f.name}"] = (self_val, other_val)
        return result

    def as_kwargs(self) -> dict[str, Any]:
        """Return a flat dict suitable for ``pyphi.config.override(**snap.as_kwargs())``.

        Field names are unique across all three layers (enforced by the
        build-time collision check in :mod:`pyphi.conf._field_routing`),
        so flattening is unambiguous.
        """
        result: dict[str, Any] = {}
        for layer_name in ("formalism", "infrastructure", "numerics"):
            layer = getattr(self, layer_name)
            for f in fields(layer):
                result[f.name] = getattr(layer, f.name)
        return result
