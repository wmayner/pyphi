"""Self-owning layered configuration global.

The :class:`_GlobalConfig` instance owns three frozen dataclass layers
(``formalism``, ``infrastructure``, ``numerics``) directly. Top-level
field writes route through :data:`FIELD_TO_LAYER` and replace the owning
layer via :func:`dataclasses.replace`, so each layer remains immutable.

Both flat (``config.precision``) and layered (``config.numerics.precision``)
forms work for reads. Writes use the flat form (``config.precision = 6``)
or :meth:`override` for scoped changes; wholesale layer replacement
(``config.numerics = NumericsConfig(precision=6)``) is also supported.

Legacy uppercase access (``config.PRECISION``) is preserved as syntax
sugar — names are case-folded and routed to the appropriate layer.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from pyphi.conf._callbacks import configure_logging
from pyphi.conf._callbacks import warn_distinction_phi_normalization_change
from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

_LAYER_NAMES = ("formalism", "infrastructure", "numerics")
_LAYER_TYPES: dict[str, type] = {
    "formalism": FormalismConfig,
    "infrastructure": InfrastructureConfig,
    "numerics": NumericsConfig,
}
_LOG_FIELDS = frozenset({"log_file", "log_file_level", "log_stdout_level"})


class _GlobalConfig:
    """Layered configuration global.

    Stores a :class:`FormalismConfig`, :class:`InfrastructureConfig`, and
    :class:`NumericsConfig` instance directly. Field writes are routed to
    the owning layer and replace it via :func:`dataclasses.replace`.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_formalism", FormalismConfig())
        object.__setattr__(self, "_infrastructure", InfrastructureConfig())
        object.__setattr__(self, "_numerics", NumericsConfig())
        infra = self._infrastructure
        configure_logging(infra.log_file, infra.log_file_level, infra.log_stdout_level)

    @property
    def formalism(self) -> FormalismConfig:
        return self._formalism

    @property
    def infrastructure(self) -> InfrastructureConfig:
        return self._infrastructure

    @property
    def numerics(self) -> NumericsConfig:
        return self._numerics

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            formalism=self._formalism,
            infrastructure=self._infrastructure,
            numerics=self._numerics,
        )

    def install_snapshot(self, snapshot: ConfigSnapshot) -> None:
        """Apply ``snapshot`` to the live global durably (not scoped).

        Worker processes call this at the start of each parallel chunk to
        seed their global config from a snapshot captured by the parent
        scheduler. Distinct from :meth:`override`, which is a scoped
        context manager.
        """
        for key, value in snapshot.as_kwargs().items():
            setattr(self, key, value)

    def override(self, **kwargs: Any) -> _OverrideContext:
        """Scoped override of one or more config fields.

        Returns a :class:`contextlib.ContextDecorator` — usable as
        ``with config.override(...):`` or ``@config.override(...)``.
        Accepts both layered lowercase names (``precision=6``) and legacy
        uppercase names (``PRECISION=6``); unknown names raise
        :class:`ConfigurationError`.
        """
        return _OverrideContext(self, kwargs)

    def load_yaml(self, path: str | Path) -> None:
        """Load a 2.0 nested-format YAML config file.

        Each layer's section is applied via per-field writes. Raises
        :class:`ConfigurationError` on unrecognized keys or 1.x flat
        format.
        """
        from pyphi.conf._io import load_yaml as _load

        data = _load(path)
        for fields_dict in data.values():
            for field_name, value in fields_dict.items():
                setattr(self, field_name, value)

    def to_yaml(self, path: str | Path) -> None:
        """Write the current config in 2.0 nested-format YAML."""
        data = {
            "formalism": asdict(self._formalism),
            "infrastructure": asdict(self._infrastructure),
            "numerics": asdict(self._numerics),
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def __dir__(self) -> list[str]:
        """Advertise leaf setting names for tab completion.

        ``pyphi.config.<TAB>`` shows leaf settings (``precision``,
        ``parallel``, ``repertoire_distance``, …) directly, so users
        don't have to memorize which setting lives in which layer.
        Also includes the layer objects themselves and the methods.
        """
        leaves = list(FIELD_TO_LAYER.keys())
        own = [
            "formalism",
            "infrastructure",
            "numerics",
            "override",
            "snapshot",
            "install_snapshot",
            "load_yaml",
            "to_yaml",
        ]
        return sorted(set(leaves + own))

    def __getattr__(self, name: str) -> Any:
        if name.isupper():
            field_name = name.lower()
            if field_name in FIELD_TO_LAYER:
                layer_name = FIELD_TO_LAYER[field_name]
                return getattr(getattr(self, "_" + layer_name), field_name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        field_name = name.lower() if name.isupper() else name

        if field_name in _LAYER_NAMES and isinstance(value, _LAYER_TYPES[field_name]):
            old_layer = getattr(self, "_" + field_name)
            object.__setattr__(self, "_" + field_name, value)
            self._fire_layer_replacement_callbacks(old_layer, value)
            return

        if field_name in FIELD_TO_LAYER:
            layer_name = FIELD_TO_LAYER[field_name]
            layer_attr = "_" + layer_name
            current_layer = getattr(self, layer_attr)
            new_layer = replace(current_layer, **{field_name: value})
            object.__setattr__(self, layer_attr, new_layer)
            self._fire_field_callback(field_name)
            return

        if field_name in _LAYER_NAMES:
            expected = _LAYER_TYPES[field_name]
            raise ConfigurationError(
                f"Cannot replace layer {field_name!r} with "
                f"{type(value).__name__}; expected {expected.__name__}."
            )
        raise ConfigurationError(
            f"Unknown config option: {name!r}. "
            "See changelog.d/p10-config-split.refactor.md for the rename map."
        )

    def _fire_field_callback(self, field_name: str) -> None:
        if field_name in _LOG_FIELDS:
            infra = self._infrastructure
            configure_logging(
                infra.log_file, infra.log_file_level, infra.log_stdout_level
            )
        elif field_name == "distinction_phi_normalization":
            warn_distinction_phi_normalization_change()

    def _fire_layer_replacement_callbacks(self, old_layer: Any, new_layer: Any) -> None:
        for f in fields(type(new_layer)):
            old_val = getattr(old_layer, f.name)
            new_val = getattr(new_layer, f.name)
            if old_val != new_val:
                self._fire_field_callback(f.name)


class _OverrideContext(contextlib.ContextDecorator):
    """Scoped override returned by :meth:`_GlobalConfig.override`.

    Saves a full snapshot on entry and restores any changed fields on
    exit. Usable as both a context manager and a decorator.
    """

    def __init__(self, config: _GlobalConfig, kwargs: dict[str, Any]) -> None:
        self._config = config
        self._new_values = kwargs
        self._saved: ConfigSnapshot | None = None

    def __enter__(self) -> _OverrideContext:
        self._saved = self._config.snapshot()
        for name, value in self._new_values.items():
            setattr(self._config, name, value)
        return self

    def __exit__(self, *exc: Any) -> bool:
        del exc
        if self._saved is None:
            return False
        saved_kwargs = self._saved.as_kwargs()
        current_kwargs = self._config.snapshot().as_kwargs()
        for name, saved_value in saved_kwargs.items():
            if current_kwargs[name] != saved_value:
                setattr(self._config, name, saved_value)
        self._saved = None
        return False
