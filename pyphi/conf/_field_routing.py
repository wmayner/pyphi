"""Build-time map from flat field name to owning layer name.

Used by :class:`pyphi.conf.legacy_global._GlobalConfig` ``__setattr__`` to
route ``config.precision = 6`` to the correct frozen layer, and by
``override(**kwargs)`` to dispatch kwargs across layers. Raises at module
import time if any field name appears in two layers — fail-fast prevents
silent misdispatch.
"""

from __future__ import annotations

from dataclasses import fields

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


class ConfigurationError(ValueError):
    """Raised on config schema problems (collisions, unknown options, etc.)."""


def _build_field_map(layers: list[tuple[str, type]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for layer_name, layer_cls in layers:
        for f in fields(layer_cls):
            if f.name in out:
                raise ConfigurationError(
                    f"Config field name collision: {f.name!r} appears in both "
                    f"{out[f.name]!r} and {layer_name!r}. Rename one."
                )
            out[f.name] = layer_name
    return out


FIELD_TO_LAYER: dict[str, str] = _build_field_map(
    [
        ("formalism", FormalismConfig),
        ("infrastructure", InfrastructureConfig),
        ("numerics", NumericsConfig),
    ]
)
