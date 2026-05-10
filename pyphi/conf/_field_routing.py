"""Build-time map from flat field name to owning (layer, sub-namespace) path.

Used by :class:`pyphi.conf._global._GlobalConfig` ``__setattr__`` and
``__getattr__`` to route ``config.precision = 6`` to the correct frozen
layer, and by ``override(**kwargs)`` to dispatch kwargs across layers.

For non-formalism layers, the sub-namespace is ``None``. For formalism,
it is ``"iit"`` or ``"actual_causation"``. Fields that collide between
``IITConfig`` and ``ActualCausationConfig`` are EXCLUDED from
:data:`FIELD_TO_LAYER` — flat writes to those names raise.
"""

from __future__ import annotations

from dataclasses import fields

from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


class ConfigurationError(ValueError):
    """Raised on config schema problems (collisions, unknown options, etc.)."""


def _build_field_map() -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    flat_layers: list[tuple[str, type]] = [
        ("infrastructure", InfrastructureConfig),
        ("numerics", NumericsConfig),
    ]
    for layer_name, layer_cls in flat_layers:
        for f in fields(layer_cls):
            if f.name in out:
                raise ConfigurationError(
                    f"Config field name collision: {f.name!r} appears in both "
                    f"{out[f.name]!r} and ({layer_name!r}, None). Rename one."
                )
            out[f.name] = (layer_name, None)

    for f in fields(FormalismConfig):
        if f.name in out:
            raise ConfigurationError(
                f"Config field name collision: {f.name!r} appears in both "
                f"{out[f.name]!r} and ('formalism', None)."
            )
        out[f.name] = ("formalism", None)

    iit_field_names = {f.name for f in fields(IITConfig)}
    ac_field_names = {f.name for f in fields(ActualCausationConfig)}

    for name in iit_field_names - ac_field_names:
        if name in out:
            raise ConfigurationError(
                f"Config field name collision: IIT field {name!r} also appears in "
                f"{out[name]!r}."
            )
        out[name] = ("formalism", "iit")

    for name in ac_field_names - iit_field_names:
        if name in out:
            raise ConfigurationError(
                f"Config field name collision: AC field {name!r} also appears in "
                f"{out[name]!r}."
            )
        out[name] = ("formalism", "actual_causation")

    return out


FIELD_TO_LAYER: dict[str, tuple[str, str | None]] = _build_field_map()


def colliding_formalism_fields() -> set[str]:
    """Field names existing in both IIT and AC sub-namespaces.

    Excluded from :data:`FIELD_TO_LAYER` — flat writes to these raise.
    """
    iit_field_names = {f.name for f in fields(IITConfig)}
    ac_field_names = {f.name for f in fields(ActualCausationConfig)}
    return iit_field_names & ac_field_names
