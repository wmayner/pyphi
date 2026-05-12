"""Self-owning layered configuration global.

The :class:`_GlobalConfig` instance owns three frozen dataclass layers
(``formalism``, ``infrastructure``, ``numerics``) directly. Top-level
field writes route through :data:`FIELD_TO_LAYER` and replace the owning
layer (or formalism sub-namespace) via :func:`dataclasses.replace`, so
each layer remains immutable.

Both flat (``config.precision``) and layered
(``config.numerics.precision``, ``config.formalism.iit.mechanism_phi_measure``)
forms work for reads. Writes use the flat form (``config.precision = 6``)
or :meth:`override` for scoped changes; wholesale layer replacement
(``config.numerics = NumericsConfig(precision=6)``) and sub-namespace
replacement (``config.iit = IITConfig(mechanism_phi_measure="EMD")``) are
also supported.

Legacy uppercase access (``config.PRECISION``) is preserved as syntax
sugar — names are case-folded and routed to the appropriate layer.

Field names that collide between the formalism's IIT and AC
sub-namespaces (currently only ``mechanism_partition_scheme``) are NOT
flat-routable; reads and writes against the bare leaf name raise. Use
the qualified path or replace the sub-namespace as a whole.
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
from pyphi.conf._field_routing import colliding_formalism_fields
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
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


def _rebuild_nested(current: Any, parts: list[str], value: Any, full_path: str) -> Any:
    """Replace a leaf field inside a frozen nested dataclass via path."""
    if not hasattr(current, parts[0]):
        raise KeyError(f"Unknown config path: {full_path!r}")
    if len(parts) == 1:
        return replace(current, **{parts[0]: value})
    sub = getattr(current, parts[0])
    new_sub = _rebuild_nested(sub, parts[1:], value, full_path)
    return replace(current, **{parts[0]: new_sub})


def _read_via_target(
    cfg: _GlobalConfig, target: tuple[str, str | None], field_name: str
) -> Any:
    """Read ``field_name`` from the (layer[, sub-namespace]) path in ``target``."""
    layer_name, sub_namespace = target
    layer = getattr(cfg, "_" + layer_name)
    if sub_namespace is None:
        return getattr(layer, field_name)
    return getattr(getattr(layer, sub_namespace), field_name)


def _write_via_target(
    cfg: _GlobalConfig,
    target: tuple[str, str | None],
    field_name: str,
    value: Any,
) -> None:
    """Replace ``field_name`` on the resolved (layer[, sub-namespace]) immutably."""
    layer_name, sub_namespace = target
    layer_attr = "_" + layer_name
    current_layer = getattr(cfg, layer_attr)
    if sub_namespace is None:
        new_layer = replace(current_layer, **{field_name: value})
    else:
        current_sub = getattr(current_layer, sub_namespace)
        new_sub = replace(current_sub, **{field_name: value})
        new_layer = replace(current_layer, **{sub_namespace: new_sub})
    object.__setattr__(cfg, layer_attr, new_layer)


class _GlobalConfig:
    """Layered configuration global.

    Stores a :class:`FormalismConfig`, :class:`InfrastructureConfig`, and
    :class:`NumericsConfig` instance directly. Field writes are routed to
    the owning layer (and sub-namespace, for formalism) and replace it
    via :func:`dataclasses.replace`.
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
        # Replace whole layers wholesale rather than streaming via
        # ``as_kwargs``: ``as_kwargs`` excludes colliding formalism field
        # names, so per-key restoration would silently lose those values.
        old_formalism = self._formalism
        object.__setattr__(self, "_formalism", snapshot.formalism)
        self._fire_layer_replacement_callbacks(old_formalism, snapshot.formalism)

        old_infra = self._infrastructure
        object.__setattr__(self, "_infrastructure", snapshot.infrastructure)
        self._fire_layer_replacement_callbacks(old_infra, snapshot.infrastructure)

        old_numerics = self._numerics
        object.__setattr__(self, "_numerics", snapshot.numerics)
        self._fire_layer_replacement_callbacks(old_numerics, snapshot.numerics)

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

        ``formalism`` is loaded from its nested ``iit`` and
        ``actual_causation`` sub-sections. Each leaf write is applied via
        :meth:`__setattr__` so the field-routing checks (unknown names,
        colliding names) fire identically to programmatic writes.
        """
        from pyphi.conf._io import load_yaml as _load

        data = _load(path)
        formalism_data = data.pop("formalism", {})
        for fields_dict in data.values():
            for field_name, value in fields_dict.items():
                setattr(self, field_name, value)
        for sub_name in ("iit", "actual_causation"):
            sub_data = formalism_data.get(sub_name, {})
            for field_name, value in sub_data.items():
                setattr(self, field_name, value)

    def to_yaml(self, path: str | Path) -> None:
        """Write the current config in 2.0 nested-format YAML."""
        with open(path, "w") as f:
            yaml.safe_dump(self._as_nested_dict(), f, sort_keys=False)

    def _as_nested_dict(self) -> dict[str, Any]:
        """Return the layered config as a nested dict."""
        return {
            "formalism": asdict(self._formalism),
            "infrastructure": asdict(self._infrastructure),
            "numerics": asdict(self._numerics),
        }

    def __getitem__(self, path: str) -> Any:
        """Read a config field by dotted path.

        ``config["numerics.precision"]``,
        ``config["formalism.iit.mechanism_phi_measure"]``,
        ``config["infrastructure.parallel"]``.
        """
        parts = path.split(".")
        if not parts or not all(parts):
            raise KeyError(f"Invalid config path: {path!r}")
        obj: Any = self
        for p in parts:
            try:
                obj = getattr(obj, p)
            except AttributeError as exc:
                raise KeyError(f"Unknown config path: {path!r}") from exc
        return obj

    def __setitem__(self, path: str, value: Any) -> None:
        """Write a config field by dotted path.

        ``config["numerics.precision"] = 6``,
        ``config["formalism.iit.mechanism_phi_measure"] = "EMD"``.
        """
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise KeyError(
                f"Path must address a leaf field on a layer "
                f"(e.g. 'numerics.precision'): {path!r}"
            )
        layer_name = parts[0]
        if layer_name not in _LAYER_NAMES:
            raise KeyError(
                f"Unknown top-level layer {layer_name!r}; expected one of {_LAYER_NAMES}"
            )
        new_layer = _rebuild_nested(
            getattr(self, "_" + layer_name), parts[1:], value, path
        )
        setattr(self, layer_name, new_layer)

    def __repr__(self) -> str:
        return yaml.safe_dump(
            self._as_nested_dict(), sort_keys=False, default_flow_style=False
        ).rstrip()

    def __dir__(self) -> list[str]:
        """Advertise leaf setting names for tab completion.

        ``pyphi.config.<TAB>`` shows leaf settings (``precision``,
        ``parallel``, ``mechanism_phi_measure``, …) directly, so users
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
                return _read_via_target(self, FIELD_TO_LAYER[field_name], field_name)
        if name in colliding_formalism_fields():
            raise AttributeError(
                f"{name!r} is ambiguous (exists in both formalism.iit and "
                "formalism.actual_causation). Use the qualified path: "
                f"config.formalism.iit.{name} or "
                f"config.formalism.actual_causation.{name}."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        field_name = name.lower() if name.isupper() else name

        # Wholesale layer replacement (e.g. config.numerics = NumericsConfig(...))
        if field_name in _LAYER_NAMES and isinstance(value, _LAYER_TYPES[field_name]):
            old_layer = getattr(self, "_" + field_name)
            object.__setattr__(self, "_" + field_name, value)
            self._fire_layer_replacement_callbacks(old_layer, value)
            return

        # Wholesale formalism sub-namespace replacement
        if field_name == "iit" and isinstance(value, IITConfig):
            old_formalism = self._formalism
            new_formalism = replace(old_formalism, iit=value)
            object.__setattr__(self, "_formalism", new_formalism)
            self._fire_layer_replacement_callbacks(old_formalism, new_formalism)
            return
        if field_name == "actual_causation" and isinstance(value, ActualCausationConfig):
            old_formalism = self._formalism
            new_formalism = replace(old_formalism, actual_causation=value)
            object.__setattr__(self, "_formalism", new_formalism)
            self._fire_layer_replacement_callbacks(old_formalism, new_formalism)
            return

        if field_name in colliding_formalism_fields():
            raise ConfigurationError(
                f"Field {field_name!r} is ambiguous (exists in both "
                "formalism.iit and formalism.actual_causation). Use the "
                "nested form: config.iit = replace(config.formalism.iit, "
                f"{field_name}=...) or config.actual_causation = "
                f"replace(config.formalism.actual_causation, {field_name}=...)."
            )

        if field_name in FIELD_TO_LAYER:
            target = FIELD_TO_LAYER[field_name]
            _write_via_target(self, target, field_name, value)
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
        # Walk leaf fields so a top-level FormalismConfig replacement
        # still surfaces nested IIT/AC field changes to the per-field
        # callbacks. Iteration is shallow for non-formalism layers
        # (whose fields are all leaves) and one level deeper for
        # formalism (iit + actual_causation).
        for f in fields(type(new_layer)):
            old_val = getattr(old_layer, f.name)
            new_val = getattr(new_layer, f.name)
            if old_val == new_val:
                continue
            if isinstance(new_val, (IITConfig, ActualCausationConfig)):
                for sub_f in fields(type(new_val)):
                    sub_old = getattr(old_val, sub_f.name)
                    sub_new = getattr(new_val, sub_f.name)
                    if sub_old != sub_new:
                        self._fire_field_callback(sub_f.name)
            else:
                self._fire_field_callback(f.name)


class _OverrideContext(contextlib.ContextDecorator):
    """Scoped override returned by :meth:`_GlobalConfig.override`.

    Saves a full snapshot on entry and restores all three layers on exit
    (wholesale, not key-by-key) so colliding sub-namespace fields still
    round-trip correctly.
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
        self._config.install_snapshot(self._saved)
        self._saved = None
        return False
