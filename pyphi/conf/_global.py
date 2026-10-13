"""Self-owning layered configuration global.

The :class:`_GlobalConfig` instance owns three frozen dataclass layers
(``formalism``, ``infrastructure``, ``numerics``) directly. Top-level
field writes route through :data:`FIELD_TO_LAYER` and replace the owning
layer (or formalism sub-namespace) via :func:`dataclasses.replace`, so
each layer remains immutable.

Both flat (``config.precision``) and layered
(``config.numerics.precision``, ``config.formalism.iit.mechanism_phi_measure``)
forms work for reads. Writes use the flat form (``config.precision = 6``),
a dotted path (``config["iit.mechanism_phi_measure"] = ...``), or
:meth:`override` for scoped changes; wholesale layer replacement
(``config.numerics = NumericsConfig(precision=6)``) and sub-namespace
replacement (``config.iit = pyphi.iit3["iit"]``) are also supported. The
layer objects themselves are read-only.

Every write runs the cross-field checks of :mod:`pyphi.conf.constraints`
(unless ``validate_config`` is off) and leaves the configuration unchanged
when they fail. :meth:`override` and :meth:`load_yaml` apply all their
values first and check the result once, so options that must agree can
change together. ``install_snapshot``, which parallel
workers call with the parent's already-checked configuration, installs a
snapshot without checking it.

Uppercase access (``config.PRECISION``) is accepted as syntax sugar —
names are case-folded and routed to the appropriate layer.

Field names that collide between the formalism's IIT and AC
sub-namespaces (e.g. ``version``, ``mechanism_partition_scheme``) are
NOT flat-routable; reads and writes against the bare leaf name raise. Use
the qualified path or replace the sub-namespace as a whole.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Literal

import yaml

from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf._field_routing import UnregisteredValueError
from pyphi.conf._field_routing import colliding_formalism_fields
from pyphi.conf._field_routing import unknown_option_hint
from pyphi.conf._field_routing import version_advice
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
_FORMALISM_SUBNAMESPACES: frozenset[str] = frozenset(
    f.name for f in fields(FormalismConfig)
)
"""Names of the formalism sub-namespaces ('iit', 'actual_causation') —
recognized as dotted-path roots that imply a leading 'formalism.'."""


def _normalize_config_path(path: str) -> str:
    """Expand a sub-namespace-rooted dotted path to its full layer path.

    ``iit.version`` -> ``formalism.iit.version``;
    ``actual_causation.alpha_measure`` ->
    ``formalism.actual_causation.alpha_measure``. Paths already rooted at a
    top-level layer (or bare leaf names) are returned unchanged.
    """
    head, sep, _rest = path.partition(".")
    if sep and head in _FORMALISM_SUBNAMESPACES:
        return "formalism." + path
    return path


def _rebuild_nested(current: Any, parts: list[str], value: Any, full_path: str) -> Any:
    """Replace a leaf field inside a frozen nested dataclass via path."""
    if not is_dataclass(current) or isinstance(current, type):
        raise ConfigurationError(
            f"Unknown config path: {full_path!r}; "
            f"{full_path.rsplit('.', len(parts))[0]!r} is an option, not a section."
        )
    if parts[0] not in {f.name for f in fields(current)}:
        raise ConfigurationError(
            f"Unknown config path: {full_path!r}. {unknown_option_hint(parts[0])}"
        )
    if len(parts) == 1:
        return replace(current, **{parts[0]: value})
    sub = getattr(current, parts[0])
    new_sub = _rebuild_nested(sub, parts[1:], value, full_path)
    return replace(current, **{parts[0]: new_sub})


def _require_section(name: str, value: Any) -> None:
    """Raise ``ConfigurationError`` unless a config-file section ``value`` is a
    mapping of options (or empty)."""
    if value is not None and not isinstance(value, Mapping):
        raise ConfigurationError(
            f"The {name!r} section of a config file must map option names to "
            f"values; got {type(value).__name__} {value!r}."
        )


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


def _iter_leaf_paths(dc_instance: Any, prefix: str) -> Iterator[str]:
    """Yield dotted leaf paths under ``prefix`` for a dataclass instance.

    For a nested dataclass field (e.g. ``FormalismConfig.iit``), recurses
    one level into the sub-dataclass. For flat layers (numerics,
    infrastructure), yields one path per field directly.
    """
    for field in fields(dc_instance):
        attr = getattr(dc_instance, field.name)
        if is_dataclass(attr) and not isinstance(attr, type):
            sub_prefix = f"{prefix}.{field.name}"
            for sub_field in fields(attr):
                yield f"{sub_prefix}.{sub_field.name}"
        else:
            yield f"{prefix}.{field.name}"


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
        context manager. The snapshot is installed as given, without the
        cross-field checks that every other write runs.
        """
        # Replace whole layers wholesale rather than streaming via
        # ``as_kwargs``: ``as_kwargs`` excludes colliding formalism field
        # names, so per-key restoration would silently lose those values.
        object.__setattr__(self, "_formalism", snapshot.formalism)
        object.__setattr__(self, "_infrastructure", snapshot.infrastructure)
        object.__setattr__(self, "_numerics", snapshot.numerics)

    def override(
        self, _paths: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> _OverrideContext:
        """Scoped override of one or more config fields.

        Returns a :class:`contextlib.ContextDecorator` — usable as
        ``with config.override(...):`` or ``@config.override(...)``.

        Accepts flat layered names (``precision=6``) and dotted paths via the
        positional mapping or kwargs (``override({"iit.version": "IIT_3_0"})`` or
        ``override(**{"iit.version": "IIT_3_0"})``). Dotted paths accept the
        sub-namespace shorthand (``iit.version``) or the full path
        (``formalism.iit.version``). Unknown names raise
        :class:`ConfigurationError`.

        Warning
        -------
        Overrides are **process-global, not thread-local**: there is one
        configuration object per process, and entering an override changes
        what *every* thread in the process reads until it exits. Code in
        another thread that computes concurrently with an active override
        silently runs under the overridden values. Do not run concurrent
        computations under different configurations in threads of one
        process; use separate processes instead. (PyPhi's own process-based
        parallel backends are unaffected — each worker process receives an
        explicit config snapshot.)

        Notes
        -----
        The context object is a plain class, not a generator: an override
        entered without holding a reference to it stays applied for the
        life of the process (garbage collection never reverts it). For a
        process-lifetime override, plain assignment
        (``config.field = value``) expresses the intent more directly.
        """
        merged: dict[str, Any] = dict(_paths) if _paths else {}
        merged.update(kwargs)
        return _OverrideContext(self, merged)

    def load_yaml(self, path: str | Path) -> None:
        """Load a nested-format YAML config file.

        Top-level ``infrastructure`` / ``numerics`` leaves are written by
        flat name. ``formalism`` is loaded from its nested ``iit`` and
        ``actual_causation`` sub-sections; each sub-namespace leaf is
        written by its qualified path (``iit.version``) so fields whose bare
        name collides between the two sub-namespaces (e.g. ``version``,
        ``mechanism_partition_scheme``) route to the sub-namespace the YAML
        nests them under.
        """
        from pyphi.conf._io import load_yaml as _load

        data = _load(path)
        # An empty section (e.g. a bare ``numerics:`` key with nothing
        # nested under it) loads from YAML as ``None``, not ``{}``; treat it
        # as an empty section rather than failing with a bare AttributeError
        # on the ``.items()`` call below.
        data = {section: (fields_dict or {}) for section, fields_dict in data.items()}
        for section, fields_dict in data.items():
            _require_section(section, fields_dict)
        formalism_data = dict(data.pop("formalism", {}))
        for sub_name in ("iit", "actual_causation"):
            _require_section(f"formalism.{sub_name}", formalism_data.get(sub_name))
        saved = self.snapshot()
        try:
            for section_name, fields_dict in data.items():
                for field_name, value in fields_dict.items():
                    layer, _ = FIELD_TO_LAYER.get(field_name, (None, None))
                    if layer is None:
                        raise ConfigurationError(
                            f"Unknown config option {field_name!r} under "
                            f"{section_name!r}. {unknown_option_hint(field_name)}"
                        )
                    if layer != section_name:
                        raise ConfigurationError(
                            f"config field {field_name!r} is nested under "
                            f"{section_name!r} but belongs to {layer!r}"
                        )
                    self._assign(field_name, value)
            for sub_name in ("iit", "actual_causation"):
                sub_data = formalism_data.pop(sub_name, {}) or {}
                for field_name, value in sub_data.items():
                    self._assign_path(f"{sub_name}.{field_name}", value)
            if formalism_data:
                raise ConfigurationError(
                    f"unrecognized keys under 'formalism': "
                    f"{sorted(formalism_data)}; expected 'iit' and/or "
                    f"'actual_causation' sub-sections"
                )
            if self.infrastructure.validate_config:
                from pyphi.conf.constraints import check_config_constraints

                check_config_constraints(self)
        except Exception:
            # Don't leave a half-loaded / invalid config installed on failure.
            self.install_snapshot(saved)
            raise

    def to_yaml(self, path: str | Path) -> None:
        """Write the current config as nested-format YAML."""
        with open(path, "w") as f:
            yaml.safe_dump(self._as_nested_dict(), f, sort_keys=False)

    def _as_nested_dict(self) -> dict[str, Any]:
        """Return the layered config as a nested dict of plain builtins.

        Immutable containers stored on the frozen layers (tuples,
        ``FrozenMap``) are converted to lists and dicts so the result is
        safe for ``yaml.safe_dump``.
        """
        from pyphi.conf._helpers import plain_builtins

        return plain_builtins(
            {
                "formalism": asdict(self._formalism),
                "infrastructure": asdict(self._infrastructure),
                "numerics": asdict(self._numerics),
            }
        )

    def __getitem__(self, path: str) -> Any:
        """Read a config field by dotted path or by bare leaf name.

        Dotted paths address layered fields:
        ``config["numerics.precision"]``,
        ``config["formalism.iit.mechanism_phi_measure"]``,
        ``config["infrastructure.parallel"]``.

        Bare leaf names route through ``FIELD_TO_LAYER`` to the owning
        layer: ``config["precision"]`` returns ``config.numerics.precision``.
        Sub-namespace-rooted paths (``config["iit.version"]``) are accepted
        as shorthand for the full path (``formalism.iit.version``).
        """
        path = _normalize_config_path(path)
        parts = path.split(".")
        if not parts or not all(parts):
            raise KeyError(f"Invalid config path: {path!r}")

        if len(parts) == 1:
            leaf = parts[0]
            if leaf in FIELD_TO_LAYER:
                return _read_via_target(self, FIELD_TO_LAYER[leaf], leaf)
            raise KeyError(f"Unknown config path: {path!r}")

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
        ``config["formalism.iit.mechanism_partition_scheme"] = "JOINT_BIPARTITION"``.
        Sub-namespace-rooted paths (``config["iit.shortcircuit_sia"] = ...``)
        are accepted as shorthand for the full path. The write is checked
        like an attribute write.
        """
        before = self.snapshot()
        self._assign_path(path, value)
        self._check(before)

    def _assign_path(self, path: str, value: Any) -> None:
        """Write a config field by dotted path without the cross-field check."""
        path = _normalize_config_path(path)
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise ConfigurationError(
                f"Path must address a leaf field on a layer "
                f"(e.g. 'numerics.precision'): {path!r}"
            )
        layer_name = parts[0]
        if layer_name not in _LAYER_NAMES:
            raise ConfigurationError(
                f"Unknown top-level layer {layer_name!r}; expected one of "
                f"{_LAYER_NAMES}. {unknown_option_hint(parts[-1])}"
            )
        new_layer = _rebuild_nested(
            getattr(self, "_" + layer_name), parts[1:], value, path
        )
        object.__setattr__(self, "_" + layer_name, new_layer)

    def _check(self, before: ConfigSnapshot) -> None:
        """Run the cross-field checks after one write; undo it if they fail.

        The write is undone whatever the checks raise. A constraint violation's
        error adds how to change options that must agree, since one
        assignment cannot.
        """
        if not self._infrastructure.validate_config:
            return
        from pyphi.conf.constraints import check_config_constraints

        try:
            check_config_constraints(self)
        except UnregisteredValueError:
            self.install_snapshot(before)
            raise
        except ConfigurationError as exc:
            attempted = self._formalism.iit.version
            self.install_snapshot(before)
            if attempted != before.formalism.iit.version:
                advice = version_advice(attempted)
            else:
                advice = (
                    "Options that must agree cannot change one assignment at "
                    "a time; pass them together to one pyphi.config.override(...) "
                    "call."
                )
            raise ConfigurationError(f"{exc} {advice}") from None
        except Exception:
            self.install_snapshot(before)
            raise

    def __iter__(self) -> Iterator[str]:
        """Yield every leaf field as a dotted path in dataclass declaration order.

        Order: numerics fields, formalism fields (iit then actual_causation
        leaves, in declaration order), infrastructure fields.
        """
        yield from _iter_leaf_paths(self._numerics, "numerics")
        yield from _iter_leaf_paths(self._formalism, "formalism")
        yield from _iter_leaf_paths(self._infrastructure, "infrastructure")

    def __contains__(self, path: object) -> bool:
        """Return True if ``path`` resolves to a leaf via ``__getitem__``."""
        if not isinstance(path, str):
            return False
        try:
            self[path]
        except KeyError:
            return False
        return True

    def __len__(self) -> int:
        """Return the number of leaf fields across all layers."""
        return sum(1 for _ in self)

    def keys(self) -> list[str]:
        """Return a list of dotted leaf paths in declaration order."""
        return list(self)

    def values(self) -> list[Any]:
        """Return a list of leaf values in declaration order."""
        return [self[k] for k in self]

    def items(self) -> list[tuple[str, Any]]:
        """Return a list of ``(path, value)`` pairs in declaration order."""
        return [(k, self[k]) for k in self]

    def get(self, path: str, default: Any = None) -> Any:
        """Return ``self[path]`` or ``default`` if the path doesn't resolve."""
        try:
            return self[path]
        except KeyError:
            return default

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
                _warn_uppercase(name)
                return _read_via_target(self, FIELD_TO_LAYER[field_name], field_name)
        if name in FIELD_TO_LAYER:
            return _read_via_target(self, FIELD_TO_LAYER[name], name)
        if name in colliding_formalism_fields():
            raise AttributeError(
                f"{name!r} is ambiguous (exists in both formalism.iit and "
                "formalism.actual_causation). Use a qualified form: "
                f"config.formalism.iit.{name}, "
                f'config["iit.{name}"], or '
                f'config["actual_causation.{name}"].'
            )
        message = f"{type(self).__name__!r} object has no attribute {name!r}"
        if not name.startswith("_"):
            message += f". {unknown_option_hint(name)}"
        raise AttributeError(message)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        before = self.snapshot()
        self._assign(name, value)
        self._check(before)

    def _assign(self, name: str, value: Any) -> None:
        """Write the option or layer ``name`` without the cross-field check."""
        field_name = name.lower() if name.isupper() else name
        if field_name != name and field_name in FIELD_TO_LAYER:
            _warn_uppercase(name, stacklevel=4)

        # Wholesale layer replacement (e.g. config.numerics = NumericsConfig(...))
        if field_name in _LAYER_NAMES and isinstance(value, _LAYER_TYPES[field_name]):
            object.__setattr__(self, "_" + field_name, value)
            return

        # Wholesale formalism sub-namespace replacement
        if field_name == "iit" and isinstance(value, IITConfig):
            from pyphi.conf.presets import _keep_strategy

            value = _keep_strategy(value, self._formalism.iit)
            new_formalism = replace(self._formalism, iit=value)
            object.__setattr__(self, "_formalism", new_formalism)
            return
        if field_name == "actual_causation" and isinstance(value, ActualCausationConfig):
            new_formalism = replace(self._formalism, actual_causation=value)
            object.__setattr__(self, "_formalism", new_formalism)
            return

        if field_name == "version":
            raise ConfigurationError(
                "Field 'version' is ambiguous (exists in both formalism.iit and "
                f"formalism.actual_causation). {version_advice(value)}"
            )
        if field_name in colliding_formalism_fields():
            raise ConfigurationError(
                f"Field {field_name!r} is ambiguous (exists in both "
                "formalism.iit and formalism.actual_causation). Qualify it: "
                f'config["iit.{field_name}"] = ... (or '
                f'"actual_causation.{field_name}"), '
                f'config.override({{"iit.{field_name}": ...}}), or set the '
                "sub-namespace wholesale via config.iit = "
                f"replace(config.formalism.iit, {field_name}=...)."
            )

        if field_name in FIELD_TO_LAYER:
            target = FIELD_TO_LAYER[field_name]
            _write_via_target(self, target, field_name, value)
            return

        if field_name in _LAYER_NAMES:
            expected = _LAYER_TYPES[field_name]
            raise ConfigurationError(
                f"Cannot replace layer {field_name!r} with "
                f"{type(value).__name__}; expected {expected.__name__}."
            )
        raise ConfigurationError(
            f"Unknown config option: {name!r}. {unknown_option_hint(name)}"
        )


def _warn_uppercase(name: str, stacklevel: int = 3) -> None:
    """Warn that ``name`` is the uppercase spelling PyPhi 1.x used."""
    warnings.warn(
        f"`{name}` is the PyPhi 1.x spelling of this option and will stop "
        f"working in a future release; use `{name.lower()}`.",
        FutureWarning,
        stacklevel=stacklevel,
    )


class _OverrideContext(contextlib.ContextDecorator):
    """Scoped override returned by :meth:`_GlobalConfig.override`.

    Saves a full snapshot on entry and restores all three layers on exit
    (wholesale, not key-by-key) so colliding sub-namespace fields still
    round-trip correctly. Snapshots are kept on a stack, so one context
    object is safely reentrant — in particular when used as a decorator on
    a recursive or reentrant function.
    """

    def __init__(self, config: _GlobalConfig, kwargs: dict[str, Any]) -> None:
        self._config = config
        self._new_values = kwargs
        self._saved: list[ConfigSnapshot] = []

    def __enter__(self) -> _OverrideContext:
        saved = self._config.snapshot()
        self._saved.append(saved)
        try:
            for name, value in self._new_values.items():
                if "." in name:
                    self._config._assign_path(name, value)
                else:
                    self._config._assign(name, value)
            if self._config.infrastructure.validate_config:
                from pyphi.conf.constraints import check_config_constraints

                try:
                    check_config_constraints(self._config)
                except UnregisteredValueError:
                    raise
                except ConfigurationError as exc:
                    # Selecting a version without its preset is the usual cause.
                    attempted = self._config.formalism.iit.version
                    if attempted == saved.formalism.iit.version:
                        raise
                    raise ConfigurationError(
                        f"{exc} {version_advice(attempted)}"
                    ) from None
        except Exception:
            # A rejected (or otherwise failed) override must not leave the
            # global config in a half-applied state, since __exit__ won't run.
            self._saved.pop()
            self._config.install_snapshot(saved)
            raise
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        del exc
        if not self._saved:
            return False
        self._config.install_snapshot(self._saved.pop())
        return False
