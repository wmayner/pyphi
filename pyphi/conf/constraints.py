"""Eager config-combination validation.

Single-field validity is enforced by each config dataclass's
``__post_init__``. This module adds the orthogonal layer of *cross-field*
constraints: combinations of individually-valid options that together compute
nonsense or a silently-different quantity. They are evaluated eagerly on
:meth:`~pyphi.conf._global._GlobalConfig.override` and ``load_yaml`` (gated by
``config.infrastructure.validate_config``) so a wrong combination fails at the
point of configuration with a :class:`~pyphi.conf.ConfigurationError` that
names the two conflicting fields and a concrete fix — rather than at compute
time, deep in the math, or not at all.

A constraint fires only on a combination that is genuinely wrong, not one
that is merely inert: every shipped preset (``iit3``, ``iit4_2023``,
``iit4_2026``) passes. An option left at a default that the active formalism
never consults is not flagged — for example, an IIT 3.0 config leaves
``system_phi_measure`` at its IIT 4.0 default, but IIT 3.0 never reads it.

The measure/version constraint reproduces the reactive
``check_measure_compatible`` boundary (each formalism's ``compatible_measures``)
eagerly, so the two cannot diverge from what the compute path enforces.

Register a constraint by appending a :class:`ConfigConstraint` to
:data:`CONFIG_CONSTRAINTS`, or with :func:`register_constraint`.

Notes
-----
``system_phi_measure="INTRINSIC_INFORMATION"`` is *not* constrained to
``IIT_4_0_2026``. The Eq. 23 cap is keyed on the measure (``applies_ii_cap``),
not the version, so ``IIT_4_0_2023`` paired with that measure applies the cap
and yields the same result as ``IIT_4_0_2026`` — a valid, if redundant,
configuration.

No EMD-precision constraint is registered. Under the POT backend
(``ot.emd2``, an exact network-simplex linear program) the EMD noise floor is
machine epsilon, and IIT 3.0 phi is stable across precision 6-13 with an
identical MIP. The ``precision: 6`` pin in the IIT 3.0 preset calibrates the
golden values; it is not a correctness requirement.

The ``system_partition_scheme_compatible_with_version`` constraint binds only
under formalisms that restrict their system partition schemes. IIT 3.0 accepts
only ``DIRECTED_BIPARTITION`` / ``DIRECTED_BIPARTITION_CUT_ONE`` (its
``sia_partitions`` raises for any other scheme), so the constraint mirrors that
boundary eagerly via the formalism's ``compatible_system_partition_schemes``.
IIT 4.0 accepts any registered scheme — a non-default scheme computes a
well-defined per-scheme phi — so it declares
``compatible_system_partition_schemes = None`` and its partition scheme is left
unconstrained.
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


# Sentinel: the formalism registry isn't importable yet (the conf package's
# bootstrap auto-load of ``pyphi_config.yml`` runs during ``pyphi.conf`` import,
# before ``pyphi.formalism`` exists). Validation is skipped in that window; every
# post-import ``override`` / ``load_yaml`` still validates.
_FORMALISM_UNAVAILABLE = object()


def _active_formalism(version: str) -> Any:
    """Return the formalism instance for ``version``.

    Returns ``None`` if ``version`` is unregistered, or
    :data:`_FORMALISM_UNAVAILABLE` if the formalism registry can't be imported
    yet (the bootstrap window; see :data:`_FORMALISM_UNAVAILABLE`). Imported
    lazily: ``pyphi.formalism`` depends on ``pyphi.conf``, so a module-level
    import would be circular.
    """
    try:
        from pyphi.formalism.base import FORMALISM_REGISTRY
    except ImportError:
        return _FORMALISM_UNAVAILABLE

    try:
        return FORMALISM_REGISTRY[version]
    except KeyError:
        return None


def _compatible_measures(version: str) -> frozenset[str] | None | object:
    """Return the active formalism's ``compatible_measures`` (or the ``None`` /
    :data:`_FORMALISM_UNAVAILABLE` sentinels from :func:`_active_formalism`)."""
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return formalism
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

    formalism = _active_formalism(version)
    fields_to_check = ["mechanism_phi_measure"]
    # Whether ``system_phi_measure`` applies is a fact about the formalism
    # (IIT 3.0 derives system phi from the CES distance and never reads it),
    # so consult its declaration rather than the version-name spelling.
    if getattr(formalism, "uses_system_phi_measure", False):
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

    # ``ces_measure`` defines Φ wherever the formalism derives system Φ from
    # the CES (directly for IIT 3.0's CES distance; as the Σφ convention for
    # IIT 4.0), so an unsupported value silently computes a different
    # quantity. Formalisms without a declaration are not constrained.
    compatible_ces = getattr(formalism, "compatible_ces_measures", None)
    if compatible_ces is not None and iit.ces_measure not in compatible_ces:
        return (
            f"formalism.iit.ces_measure={iit.ces_measure!r} is not compatible "
            f"with formalism.iit.version={version!r}. Compatible CES measures "
            f"for this version: {sorted(compatible_ces)}. Fix: set "
            f"formalism.iit.ces_measure to one of those, or change "
            f"formalism.iit.version to one that supports {iit.ces_measure!r}."
        )
    return None


@register_constraint("system_partition_scheme_compatible_with_version")
def _system_partition_scheme_compatible_with_version(config: Any) -> str | None:
    """The system partition scheme must be one the active formalism accepts.

    IIT 3.0 only supports ``DIRECTED_BIPARTITION`` /
    ``DIRECTED_BIPARTITION_CUT_ONE`` system schemes (its ``sia_partitions``
    raises otherwise); pairing it with any other scheme computes nothing usable.
    Formalisms that accept any registered scheme declare
    ``compatible_system_partition_schemes = None`` and are not constrained.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        # Bootstrap window, or unregistered version (the measure constraint
        # reports an unregistered version).
        return None
    compatible = getattr(formalism, "compatible_system_partition_schemes", None)
    if compatible is None:
        return None  # unconstrained (e.g. IIT 4.0)
    scheme = iit.system_partition_scheme
    if scheme not in compatible:
        return (
            f"formalism.iit.system_partition_scheme={scheme!r} is not compatible "
            f"with formalism.iit.version={version!r}. Compatible system partition "
            f"schemes for this version: {sorted(compatible)}. Fix: set "
            f"formalism.iit.system_partition_scheme to one of those, or change "
            f"formalism.iit.version to one whose formalism accepts {scheme!r}."
        )
    return None


@register_constraint("mechanism_partition_scheme_compatible_with_version")
def _mechanism_partition_scheme_compatible_with_version(config: Any) -> str | None:
    """The mechanism partition scheme must be one the active formalism accepts.

    IIT 3.0 defines mechanism-level φ over bipartitions (with the wedge
    tripartition as its registered variant); pairing it with the IIT 4.0
    ``JOINT_PARTITION_ALL`` family silently computes a different quantity
    than the 2014 paper's φ. Formalisms that accept any registered scheme
    declare ``compatible_mechanism_partition_schemes = None`` and are not
    constrained.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return None
    compatible = getattr(formalism, "compatible_mechanism_partition_schemes", None)
    if compatible is None:
        return None  # unconstrained (e.g. IIT 4.0)
    scheme = iit.mechanism_partition_scheme
    if scheme not in compatible:
        return (
            f"formalism.iit.mechanism_partition_scheme={scheme!r} is not "
            f"compatible with formalism.iit.version={version!r}. Compatible "
            f"mechanism partition schemes for this version: "
            f"{sorted(compatible)}. Fix: set "
            f"formalism.iit.mechanism_partition_scheme to one of those, or "
            f"change formalism.iit.version to one whose formalism accepts "
            f"{scheme!r}."
        )
    return None


@register_constraint("sia_tie_resolution_compatible_with_version")
def _sia_tie_resolution_compatible_with_version(config: Any) -> str | None:
    """The SIA tie-resolution strategies must be ones the active formalism's
    SIA result type supports.

    IIT 3.0 SIA results carry only raw phi and the MIP, so strategies reading
    ``normalized_phi`` or ``purview`` (e.g. the IIT 4.0 default
    ``NORMALIZED_PHI``) raise ``AttributeError`` at compute time. Formalisms
    whose SIA type supports every registered strategy declare
    ``compatible_sia_tie_strategies = None`` and are not constrained.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return None
    compatible = getattr(formalism, "compatible_sia_tie_strategies", None)
    if compatible is None:
        return None  # unconstrained (e.g. IIT 4.0)
    strategy = iit.sia_tie_resolution
    components = (strategy,) if isinstance(strategy, str) else tuple(strategy)
    for component in components:
        if component not in compatible:
            return (
                f"formalism.iit.sia_tie_resolution component {component!r} is "
                f"not compatible with formalism.iit.version={version!r}. "
                f"Compatible SIA tie strategies for this version: "
                f"{sorted(compatible)}. Fix: set formalism.iit.sia_tie_resolution "
                f"to use only those (the shipped preset uses "
                f"['PHI', 'PARTITION_LEX']), or change formalism.iit.version."
            )
    return None
