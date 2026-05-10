"""Frozen snapshot of the three config layers, attached to result objects.

A :class:`ConfigSnapshot` mirrors the live ``pyphi.config`` shape but is
immutable: once a result object carries a snapshot, mutating the live
global doesn't change the snapshot's view of what produced the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

from pyphi.conf._field_routing import colliding_formalism_fields
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
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
        """Return ``{dotted-path: (self_value, other_value)}`` for fields that differ.

        Walks one level deeper into ``formalism`` so nested IIT and AC
        sub-namespace fields surface with their qualified paths
        (``formalism.iit.repertoire_measure``, ...).
        """
        result: dict[str, tuple[Any, Any]] = {}
        for layer_name in ("infrastructure", "numerics"):
            self_layer = getattr(self, layer_name)
            other_layer = getattr(other, layer_name)
            for f in fields(self_layer):
                self_val = getattr(self_layer, f.name)
                other_val = getattr(other_layer, f.name)
                if self_val != other_val:
                    result[f"{layer_name}.{f.name}"] = (self_val, other_val)
        for sub_name in ("iit", "actual_causation"):
            self_sub = getattr(self.formalism, sub_name)
            other_sub = getattr(other.formalism, sub_name)
            for f in fields(self_sub):
                self_val = getattr(self_sub, f.name)
                other_val = getattr(other_sub, f.name)
                if self_val != other_val:
                    result[f"formalism.{sub_name}.{f.name}"] = (self_val, other_val)
        return result

    def as_kwargs(self) -> dict[str, Any]:
        """Return a flat dict suitable for ``pyphi.config.override(**snap.as_kwargs())``.

        Field names that collide between the formalism's IIT and AC
        sub-namespaces (currently only ``mechanism_partition_scheme``)
        are excluded — flat overrides on those names are ambiguous and
        :class:`pyphi.conf._global._GlobalConfig.__setattr__` rejects
        them. To round-trip a colliding-name change, use sub-namespace
        wholesale replacement (``config.iit = ...``).
        """
        result: dict[str, Any] = {}
        for layer in (self.infrastructure, self.numerics):
            for f in fields(layer):
                result[f.name] = getattr(layer, f.name)
        excluded = colliding_formalism_fields()
        for sub_name in ("iit", "actual_causation"):
            sub_layer = getattr(self.formalism, sub_name)
            for f in fields(sub_layer):
                if f.name in excluded:
                    continue
                result[f.name] = getattr(sub_layer, f.name)
        return result


__all__ = [
    "ActualCausationConfig",
    "ConfigSnapshot",
    "FormalismConfig",
    "IITConfig",
    "InfrastructureConfig",
    "NumericsConfig",
]
