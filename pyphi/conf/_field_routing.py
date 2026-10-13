"""Build-time map from flat field name to owning (layer, sub-namespace) path.

Used by :class:`pyphi.conf._global._GlobalConfig` ``__setattr__`` and
``__getattr__`` to route ``config.precision = 6`` to the correct frozen
layer, and by ``override(**kwargs)`` to dispatch kwargs across layers.

For non-formalism layers, the sub-namespace is ``None``. For formalism,
it is ``"iit"`` or ``"actual_causation"``. Fields that collide between
``IITConfig`` and ``ActualCausationConfig`` are EXCLUDED from
:data:`FIELD_TO_LAYER` — flat writes to those names raise.

The module also records former option names (:data:`RENAMED_FIELDS`,
:data:`REMOVED_FIELDS`) and builds the error messages for unknown options and
for assignments to the read-only layers, so each says how to write the option.
"""

from __future__ import annotations

import difflib
from dataclasses import fields
from typing import Any

from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.constants import MIGRATION_GUIDE_URL


class ConfigurationError(ValueError):
    """Raised on config schema problems (collisions, unknown options, etc.)."""


class UnregisteredValueError(ConfigurationError):
    """A configured name that nothing is registered under, such as a misspelt
    measure. No change to the other options makes it valid, so its message
    carries no advice about options that must agree."""


_MEASURES = ("formalism.iit.mechanism_phi_measure", "formalism.iit.system_phi_measure")

RENAMED_FIELDS: dict[str, tuple[str, ...]] = {
    "iit_version": ("formalism.iit.version",),
    "measure": _MEASURES,
    "repertoire_distance": _MEASURES,
    "repertoire_distance_specification": ("formalism.iit.specification_measure",),
    "ces_distance": ("formalism.iit.ces_measure",),
    "actual_causation_measure": ("formalism.actual_causation.alpha_measure",),
    "partition_type": ("formalism.iit.mechanism_partition_scheme",),
    "system_partition_type": ("formalism.iit.system_partition_scheme",),
    "system_partition_include_complete": (
        "formalism.iit.system_partition_include_total",
    ),
    "pick_smallest_purview": ("formalism.iit.purview_tie_resolution",),
    "assume_cuts_cannot_create_new_concepts": (
        "formalism.iit.assume_partitions_cannot_create_new_concepts",
    ),
    "parallel_cut_evaluation": ("infrastructure.parallel_partition_evaluation",),
    "parallel_concept_evaluation": ("infrastructure.parallel_distinction_evaluation",),
    "maximum_cache_memory_percentage": ("infrastructure.memory_ceiling_percentage",),
    "maximum_cache_memory_bytes": ("infrastructure.memory_ceiling_bytes",),
    "clear_subsystem_caches_after_computing_sia": (
        "infrastructure.clear_system_caches_after_computing_sia",
    ),
    "validate_subsystem_states": ("infrastructure.validate_system_states",),
}
"""Former option names, lowercased, mapped to the dotted paths of the options
that replace them.

Covers every renamed option in the migration guide's table of 1.x options and
the options renamed during the 2.0 release candidates. Options whose name only
changed case are not listed: their uppercase spelling still works.
"""

REMOVED_FIELDS: dict[str, str] = {
    "repertoire_distance_differentiation": (
        "intrinsic differentiation is the surprisal of the specified state "
        "and has no measure option"
    ),
    "system_cuts": (
        "the iit3 preset sets system_partition_scheme='DIRECTED_BIPARTITION', "
        "the IIT 3.0 cut"
    ),
    "redis_cache": "results persist through infrastructure.disk_cache_results",
    "redis_config": "results persist through infrastructure.disk_cache_results",
    "ray_config": "",
    "log_file": "call pyphi.enable_logging(level, file)",
    "log_file_level": "call pyphi.enable_logging(level, file)",
    "log_stdout_level": "call pyphi.enable_logging(level, file)",
    "validate_json_version": "",
}
"""Former option names, lowercased, that have no replacement option, mapped to
what to do instead (empty when there is nothing to do)."""


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


def layer_path(field_name: str) -> str:
    """Return the dotted path of the routable option ``field_name``.

    ``"precision"`` gives ``"numerics.precision"``; ``"shortcircuit_sia"``
    gives ``"formalism.iit.shortcircuit_sia"``.
    """
    layer, sub_namespace = FIELD_TO_LAYER[field_name]
    return ".".join(p for p in (layer, sub_namespace, field_name) if p)


def former_option_location(name: str) -> str | None:
    """Describe where the PyPhi 1.x option ``name`` is now, or return None.

    Covers options that only changed case (``PRECISION`` is
    ``'numerics.precision'``), renamed options and removed ones.
    """
    key = name.lower()
    if key in FIELD_TO_LAYER:
        return repr(layer_path(key))
    if key in RENAMED_FIELDS:
        return " and ".join(map(repr, RENAMED_FIELDS[key]))
    if key in REMOVED_FIELDS:
        reason = REMOVED_FIELDS[key]
        return f"removed ({reason})" if reason else "removed"
    return None


def version_advice(version: Any = None) -> str:
    """Say how to select ``version`` of IIT: with its preset, never alone.

    The example names the preset for ``version`` when it is a known version
    name, and ``iit4_2023`` otherwise.
    """
    from pyphi.conf.presets import _ALIASES

    preset = next(
        (alias for alias, name in _ALIASES.items() if name == version), "iit4_2023"
    )
    return (
        "To select a version of IIT, apply its preset, which also sets the "
        "measures and schemes that version requires: "
        f"pyphi.config.override(**pyphi.{preset}) for a block of code, or "
        f"pyphi.analyze(..., formalism={_ALIASES[preset]!r}) for one call."
    )


def unknown_option_hint(name: str) -> str:
    """Return a sentence for an option name the config does not have.

    Names the replacement of a former option, says that a removed option is
    gone, suggests the closest current option for a typo, and otherwise
    points to the migration guide.
    """
    key = name.lower()
    if key in RENAMED_FIELDS:
        paths = RENAMED_FIELDS[key]
        hint = f"It is now {' and '.join(map(repr, paths))}."
        if "formalism.iit.version" in paths:
            hint += " " + version_advice()
        return hint
    if key in REMOVED_FIELDS:
        reason = REMOVED_FIELDS[key]
        return f"PyPhi 2.0 removed it: {reason}." if reason else "PyPhi 2.0 removed it."
    candidates = sorted(set(FIELD_TO_LAYER) | colliding_formalism_fields())
    close = difflib.get_close_matches(key, candidates, n=1)
    if close:
        return f"Did you mean {close[0]!r}?"
    return f"See the 2.0 migration guide ({MIGRATION_GUIDE_URL}) for the rename map."


def write_hint(path: str, value: Any) -> str:
    """Return how to set the option at dotted ``path`` on ``pyphi.config``."""
    if path == "formalism.iit.version":
        return version_advice(value)
    name = path.rsplit(".", 1)[-1]
    shown = repr(value) if isinstance(value, str | int | float | bool | None) else "..."
    if name in FIELD_TO_LAYER:
        return (
            f"Set the option with pyphi.config.{name} = {shown}, or "
            f"pyphi.config.override({name}={shown}) for a block of code."
        )
    key = path.removeprefix("formalism.")
    return (
        f'Set the option with pyphi.config["{key}"] = {shown}, or '
        f'pyphi.config.override({{"{key}": {shown}}}) for a block of code.'
    )


_LAYER_PATHS: dict[type, str] = {
    NumericsConfig: "numerics",
    InfrastructureConfig: "infrastructure",
    FormalismConfig: "formalism",
    IITConfig: "formalism.iit",
    ActualCausationConfig: "formalism.actual_causation",
}


def layer_write_error(layer: Any, name: str, value: Any) -> ConfigurationError:
    """Return the error for assigning ``value`` to ``name`` on a frozen layer.

    The layers are immutable; the message shows how to change the option on
    ``pyphi.config`` instead.
    """
    cls_name = type(layer).__name__
    if name not in {f.name for f in fields(layer)}:
        return ConfigurationError(
            f"{cls_name} has no option {name!r}. {unknown_option_hint(name)}"
        )
    return ConfigurationError(
        f"{cls_name} objects are read-only. "
        f"{write_hint(f'{_LAYER_PATHS[type(layer)]}.{name}', value)} For a "
        f"modified copy of this object, use dataclasses.replace(obj, {name}=...)."
    )
