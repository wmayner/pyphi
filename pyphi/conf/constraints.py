"""Eager config-combination validation (roadmap B13).

Single-field validity is enforced by each config dataclass's
``__post_init__``. This module adds the orthogonal layer of *cross-field*
constraints: combinations of individually-valid options that together compute
nonsense or a silently-different quantity. They are evaluated eagerly on
:meth:`~pyphi.conf._global._GlobalConfig.override` and ``load_yaml`` (gated by
``config.infrastructure.validate_config``) so a wrong combination fails at the
point of configuration with a :class:`~pyphi.conf.ConfigurationError` that
names the two conflicting fields and a concrete fix — rather than at compute
time, deep in the math, or not at all.

Design notes
------------

- **Conservative by construction.** Every shipped preset (``iit3``,
  ``iit4_2023``, ``iit4_2026``) must pass, and a constraint is only added when
  the combination is genuinely wrong with code evidence — not merely *inert*.
  For example, an IIT-3.0 config leaves ``system_phi_measure`` at its 4.0
  default, but IIT 3.0 never consults it, so that is not flagged.

- **Mirrors the established source of truth.** The measure/version constraints
  reproduce the existing reactive ``check_measure_compatible`` boundary (each
  formalism's ``compatible_measures``) but eagerly, so they cannot diverge from
  what the compute path already enforces.

- **Verified, not assumed.** Only combinations confirmed to be wrong are
  encoded. Notably, ``system_phi_measure="INTRINSIC_INFORMATION"`` is *not*
  constrained to ``IIT_4_0_2026``: the Eq. 23 cap is keyed on the measure
  (``applies_ii_cap``), not the version, so ``IIT_4_0_2023`` + that measure
  correctly applies the cap (it equals the 2026 result, confirmed
  empirically) — a valid, if redundant, configuration.

Registering more constraints is a matter of appending a
:class:`ConfigConstraint` to :data:`CONFIG_CONSTRAINTS` (or using
:func:`register_constraint`); add only ones backed by a confirmation
experiment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyphi.conf._field_routing import ConfigurationError

# A constraint inspects the (post-override) config and returns an error message
# — naming both conflicting fields and a fix — when violated, else ``None``.
ConstraintCheck = Callable[[Any], str | None]


@dataclass(frozen=True)
class ConfigConstraint:
    """A named cross-field config constraint."""

    name: str
    check: ConstraintCheck


CONFIG_CONSTRAINTS: list[ConfigConstraint] = []


def register_constraint(name: str) -> Callable[[ConstraintCheck], ConstraintCheck]:
    """Decorator registering a constraint-check function under ``name``."""

    def decorator(func: ConstraintCheck) -> ConstraintCheck:
        CONFIG_CONSTRAINTS.append(ConfigConstraint(name=name, check=func))
        return func

    return decorator


def check_config_constraints(config: Any) -> None:
    """Run every registered constraint against ``config``.

    Raises :class:`~pyphi.conf.ConfigurationError` on the first violation, with
    a message naming the two conflicting fields and a concrete fix.
    """
    for constraint in CONFIG_CONSTRAINTS:
        message = constraint.check(config)
        if message is not None:
            raise ConfigurationError(message)


# Set of IIT versions whose formalism consults ``system_phi_measure``. IIT 3.0
# computes system phi from the cause-effect-structure distance and never reads
# this field, so it is checked only for the 4.0 family (matching the reactive
# ``check_measure_compatible`` call sites).
_VERSIONS_USING_SYSTEM_MEASURE = ("IIT_4_0",)


# Sentinel: the formalism registry isn't importable yet (the conf package's
# bootstrap auto-load of ``pyphi_config.yml`` runs during ``pyphi.conf`` import,
# before ``pyphi.formalism`` exists). Validation is skipped in that window; every
# post-import ``override`` / ``load_yaml`` still validates.
_FORMALISM_UNAVAILABLE = object()


def _compatible_measures(version: str) -> frozenset[str] | None | object:
    """Return the active formalism's ``compatible_measures``.

    Returns ``None`` if ``version`` is unregistered, or
    :data:`_FORMALISM_UNAVAILABLE` if the formalism registry can't be imported
    yet. Imported lazily: ``pyphi.formalism`` depends on ``pyphi.conf``, so a
    module-level import would be circular.
    """
    try:
        from pyphi.formalism.base import FORMALISM_REGISTRY
    except ImportError:
        return _FORMALISM_UNAVAILABLE

    try:
        formalism = FORMALISM_REGISTRY[version]
    except KeyError:
        return None
    return frozenset(formalism.compatible_measures)


@register_constraint("measure_compatible_with_version")
def _measure_compatible_with_version(config: Any) -> str | None:
    """The configured measures must be defined by the active IIT formalism.

    Pairing a version with a measure outside its ``compatible_measures`` (e.g.
    ``IIT_3_0`` with ``INTRINSIC_INFORMATION``, or ``IIT_4_0_2023`` with
    ``EMD``) computes a different mathematical object than that formalism's φ.
    """
    iit = config.formalism.iit
    version = iit.version

    compatible = _compatible_measures(version)
    if compatible is _FORMALISM_UNAVAILABLE:
        return None  # bootstrap window; see _FORMALISM_UNAVAILABLE
    if compatible is None:
        from pyphi.formalism.base import FORMALISM_REGISTRY

        return (
            f"formalism.iit.version={version!r} is not a registered IIT "
            f"formalism. Fix: set formalism.iit.version to one of "
            f"{sorted(FORMALISM_REGISTRY.store)}."
        )
    assert isinstance(compatible, frozenset)

    fields_to_check = ["mechanism_phi_measure"]
    if version.startswith(_VERSIONS_USING_SYSTEM_MEASURE):
        fields_to_check.append("system_phi_measure")

    for field_name in fields_to_check:
        measure = getattr(iit, field_name)
        if measure not in compatible:
            return (
                f"formalism.iit.{field_name}={measure!r} is not compatible "
                f"with formalism.iit.version={version!r}. Compatible measures "
                f"for this version: {sorted(compatible)}. Fix: set "
                f"formalism.iit.{field_name} to one of those, or change "
                f"formalism.iit.version to one whose formalism defines "
                f"{measure!r}."
            )
    return None
