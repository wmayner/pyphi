"""Frozen snapshot of the three config layers, attached to result objects.

A :class:`ConfigSnapshot` mirrors the live ``pyphi.config`` shape but is
immutable: once a result object carries a snapshot, mutating the live
global doesn't change the snapshot's view of what produced the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

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
    ``pyphi.config.override(**snap.as_overrides())``, which restores every
    field — including the formalism version and other fields whose bare
    names collide between the IIT and actual-causation sub-namespaces.
    """

    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def diff(self, other: ConfigSnapshot) -> dict[str, tuple[Any, Any]]:
        """Return ``{dotted-path: (self_value, other_value)}`` for fields that differ.

        Walks one level deeper into ``formalism`` so nested IIT and AC
        sub-namespace fields surface with their qualified paths
        (``formalism.iit.mechanism_phi_measure``, ...).
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

    def as_overrides(self) -> dict[str, Any]:
        """Return a full-fidelity override mapping for this snapshot.

        Infrastructure and numerics fields appear under their flat names;
        every formalism field appears under its qualified dotted path
        (``iit.version``, ``actual_causation.mechanism_partition_scheme``,
        ...), so — unlike :meth:`as_kwargs` — fields whose bare names
        collide between the two formalism sub-namespaces are included.
        ``pyphi.config.override(**snap.as_overrides())`` reproduces the
        snapshotted configuration exactly.
        """
        result: dict[str, Any] = {}
        for layer in (self.infrastructure, self.numerics):
            for f in fields(layer):
                result[f.name] = getattr(layer, f.name)
        for sub_name in ("iit", "actual_causation"):
            sub = getattr(self.formalism, sub_name)
            for f in fields(sub):
                result[f"{sub_name}.{f.name}"] = getattr(sub, f.name)
        return result

    @classmethod
    def from_builtins(cls, data: Mapping[str, Any]) -> ConfigSnapshot:
        """Rehydrate a snapshot from its plain-builtins (dict) form.

        Inverse of ``msgspec.to_builtins`` on a snapshot, as used by the
        serialization layer. Unknown field names are ignored and missing
        fields take their defaults, so payloads written by other PyPhi
        versions still load.
        """

        def build(layer_cls: type, layer_data: Mapping[str, Any]) -> Any:
            names = {f.name for f in fields(layer_cls)}
            return layer_cls(**{k: v for k, v in layer_data.items() if k in names})

        formalism_data = data.get("formalism", {})
        return cls(
            formalism=FormalismConfig(
                iit=build(IITConfig, formalism_data.get("iit", {})),
                actual_causation=build(
                    ActualCausationConfig, formalism_data.get("actual_causation", {})
                ),
            ),
            infrastructure=build(InfrastructureConfig, data.get("infrastructure", {})),
            numerics=build(NumericsConfig, data.get("numerics", {})),
        )

    def as_kwargs(self) -> dict[str, Any]:
        """Return a flat dict suitable for ``pyphi.config.override(**snap.as_kwargs())``.

        Field names that collide between the formalism's IIT and AC
        sub-namespaces (e.g. ``version``, ``mechanism_partition_scheme``)
        are excluded by :meth:`FormalismConfig.as_kwargs` — flat overrides
        on those names are ambiguous and
        :class:`pyphi.conf._global._GlobalConfig.__setattr__` rejects them.
        To round-trip a colliding-name change, use sub-namespace wholesale
        replacement (``config.iit = ...``).
        """
        result: dict[str, Any] = {}
        for layer in (self.infrastructure, self.numerics):
            for f in fields(layer):
                result[f.name] = getattr(layer, f.name)
        result.update(self.formalism.as_kwargs())
        return result


__all__ = [
    "ActualCausationConfig",
    "ConfigSnapshot",
    "FormalismConfig",
    "IITConfig",
    "InfrastructureConfig",
    "NumericsConfig",
]
