"""Eager config-combination validation.

Single-field validity is enforced by each config dataclass's
``__post_init__``. This module adds the orthogonal layer of *cross-field*
constraints: combinations of individually-valid options that together compute
nonsense or a silently-different quantity. They are evaluated on every write
to ``pyphi.config``: after each assignment, and once after all the values of an
:meth:`~pyphi.conf._global._GlobalConfig.override` or ``load_yaml`` call are
applied (gated by ``config.infrastructure.validate_config``). A wrong
combination fails at the point of configuration with a
:class:`~pyphi.conf.ConfigurationError` that names the two conflicting fields
and a concrete fix, rather than at compute time, deep in the math, or not at
all. Snapshots installed in parallel workers come from an already-checked
configuration and are not checked again.

A constraint fires only on a combination that is wrong, not one that is
merely inert. An option left at a default that the active formalism
never consults is not flagged — for example, an IIT 3.0 config leaves
``system_phi_measure`` at its IIT 4.0 default, but IIT 3.0 never reads it.

The measure/version constraint reproduces the reactive
``check_measure_compatible`` boundary (each formalism's ``compatible_measures``)
eagerly, so the two cannot diverge from what the compute path enforces.

Register a constraint by appending a :class:`ConfigConstraint` to
:data:`CONFIG_CONSTRAINTS`, or with :func:`register_constraint`. A check
returns a message for a combination that is wrong, and raises
:class:`~pyphi.conf._field_routing.UnregisteredValueError` for a name that
nothing is registered under (a misspelt measure, scheme or strategy), which no
other option can fix.

Notes
-----
The intrinsic-information requirement (Eq. 23) is keyed on the system
measure (``applies_intrinsic_information_requirement``), and each IIT 4.0
formalism declares whether it is defined by the requirement. The two must
agree. ``IIT_4_0_2026`` with a measure that lacks the requirement would compute
the 2023 quantity while reporting version 2026; ``IIT_4_0_2023`` with a
measure that applies it would compute the 2026 quantity while reporting
version 2023. Both are formalism mixtures matching no paper. The second is
also what setting ``formalism.iit.version`` alone produces, since the default
system measure applies the requirement.

The ``background_conditioning_compatible_with_version`` constraint pins IIT 3.0 to
``CONDITION_CURRENT_STATE`` (the shipped preset's convention). The marginalized IIT 3.0
variant opts out via ``validate_config=False``, which also disables the matching
dispatch-boundary checks.

``precision`` is not constrained against any other field.

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

import difflib
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf._field_routing import UnregisteredValueError

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


def _strategy_components(strategy: Any) -> tuple[str, ...] | None:
    """Return the component names of a tie-resolution strategy value.

    A strategy is one registered name or a sequence of them. Returns ``None``
    for any other value.
    """
    if isinstance(strategy, str):
        return (strategy,)
    if isinstance(strategy, Sequence) and all(isinstance(c, str) for c in strategy):
        return tuple(strategy)
    return None


def _require_registered_partition_scheme(
    path: str, scheme: Any, registry_name: str
) -> None:
    """Raise :class:`UnregisteredValueError` if ``scheme`` is not a registered
    scheme name.

    ``registry_name`` names a partition-scheme registry in
    :mod:`pyphi.partition`, imported lazily because ``pyphi.partition`` depends
    on ``pyphi.conf``. The check is skipped while that module cannot be
    imported yet (the bootstrap window).
    """
    try:
        from pyphi import partition
    except ImportError:
        return
    registered = getattr(partition, registry_name).store
    _require_registered(f"formalism.{path}", scheme, registered, "partition scheme")


def _require_registered(
    path: str,
    value: Any,
    known: Any,
    kind: str,
    *,
    component: bool = False,
    accepted_by: str | None = None,
) -> None:
    """Raise :class:`UnregisteredValueError` if ``value`` is not in ``known``.

    ``component`` marks ``value`` as one entry of the option at ``path``. The
    message suggests the closest known name. ``accepted_by`` names the
    formalism that fixes ``known``; the message then says ``value`` is not
    accepted by it, which also covers a measure that is registered but
    belongs to another formalism.
    """
    if isinstance(value, str) and value in known:
        return
    close = difflib.get_close_matches(str(value), sorted(known), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    subject = f"{path} entry {value!r}" if component else f"{path}={value!r}"
    if accepted_by is not None:
        raise UnregisteredValueError(
            f"{subject} is not one of the {kind} that {accepted_by} accepts."
            f"{hint} Accepted: {sorted(known)}."
        )
    raise UnregisteredValueError(
        f"{subject} is not a registered {kind}.{hint} Registered: {sorted(known)}."
    )


def _require_registered_measure(field_name: str, value: Any) -> None:
    """Raise :class:`UnregisteredValueError` if no measure is registered under
    ``value`` for the IIT option ``field_name``.

    A name that is not registered cannot be fixed by changing the version, so
    it is reported as unregistered rather than as incompatible. Registries are
    imported lazily: :mod:`pyphi.measures` depends on :mod:`pyphi.conf`.
    """
    from pyphi.measures import ces
    from pyphi.measures import distribution as d

    if field_name == "system_phi_measure":
        known = set(d.composite_measures)
    elif field_name == "ces_measure":
        known = set(ces.measures)
    else:
        known = (
            set(d.distribution_measures)
            | set(d.state_aware_measures)
            | set(d.stateful_distribution_measures)
            | set(d.composite_measures)
        )
    _require_registered(f"formalism.iit.{field_name}", value, known, "measure")


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
            _require_registered_measure(field_name, measure)
            return (
                f"formalism.iit.{field_name}={measure!r} is not compatible "
                f"with formalism.iit.version={version!r}. Compatible measures "
                f"for this version: {sorted(compatible)}. Fix: set "
                f"formalism.iit.{field_name} to one of those, or change "
                f"formalism.iit.version to one whose formalism defines "
                f"{measure!r}."
            )

    # ``specification_measure`` drives the specified-state search wherever
    # the formalism has one (the IIT 4.0 family; IIT 3.0 never consults it),
    # so an unsupported value silently changes Φ and φ_s. Formalisms without
    # a declaration are not constrained.
    compatible_spec = getattr(formalism, "compatible_specification_measures", None)
    if compatible_spec is not None and iit.specification_measure not in compatible_spec:
        _require_registered_measure("specification_measure", iit.specification_measure)
        return (
            f"formalism.iit.specification_measure="
            f"{iit.specification_measure!r} is not compatible with "
            f"formalism.iit.version={version!r}. Compatible specification "
            f"measures for this version: {sorted(compatible_spec)}. Fix: set "
            f"formalism.iit.specification_measure to one of those, or change "
            f"formalism.iit.version to one whose formalism accepts "
            f"{iit.specification_measure!r}."
        )

    # ``ces_measure`` defines Φ wherever the formalism derives system Φ from
    # the CES (directly for IIT 3.0's CES distance; as the Σφ convention for
    # IIT 4.0), so an unsupported value silently computes a different
    # quantity. Formalisms without a declaration are not constrained.
    compatible_ces = getattr(formalism, "compatible_ces_measures", None)
    if compatible_ces is not None and iit.ces_measure not in compatible_ces:
        _require_registered_measure("ces_measure", iit.ces_measure)
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
    """The system partition scheme must be registered and be one the active
    formalism accepts.

    IIT 3.0 only supports ``DIRECTED_BIPARTITION`` /
    ``DIRECTED_BIPARTITION_CUT_ONE`` system schemes (its ``sia_partitions``
    raises otherwise); pairing it with any other scheme computes nothing usable.
    Formalisms that accept any registered scheme declare
    ``compatible_system_partition_schemes = None`` and are not constrained.
    """
    iit = config.formalism.iit
    _require_registered_partition_scheme(
        "iit.system_partition_scheme",
        iit.system_partition_scheme,
        "system_partition_types",
    )
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
    """The mechanism partition schemes must be registered, and the IIT one
    must be one the active formalism accepts.

    The actual-causation scheme is checked for registration only. IIT 3.0
    defines mechanism-level φ over bipartitions (with the wedge
    tripartition as its registered variant); pairing it with the IIT 4.0
    ``JOINT_PARTITION_ALL`` family silently computes a different quantity
    than the 2014 paper's φ. Formalisms that accept any registered scheme
    declare ``compatible_mechanism_partition_schemes = None`` and are not
    constrained.
    """
    iit = config.formalism.iit
    for path, scheme in (
        ("iit.mechanism_partition_scheme", iit.mechanism_partition_scheme),
        (
            "actual_causation.mechanism_partition_scheme",
            config.formalism.actual_causation.mechanism_partition_scheme,
        ),
    ):
        _require_registered_partition_scheme(path, scheme, "partition_types")
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
    components = _strategy_components(iit.sia_tie_resolution)
    if components is None:
        return None  # reported by tie_resolution_strategies_registered
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


@register_constraint("tie_resolution_strategies_registered")
def _tie_resolution_strategies_registered(config: Any) -> str | None:
    """Every tie-resolution strategy component must name a registered
    strategy, so a typo fails at configuration time rather than in the
    middle of a computation.

    Imported lazily: ``pyphi.resolve_ties`` depends on ``pyphi.conf``, so a
    module-level import would be circular; during the bootstrap window the
    check is skipped (every post-import ``override`` / ``load_yaml`` still
    validates).
    """
    try:
        from pyphi.resolve_ties import phi_object_tie_resolution_strategies
    except ImportError:
        return None
    valid = set(phi_object_tie_resolution_strategies.store)
    iit = config.formalism.iit
    for field_name in (
        "state_tie_resolution",
        "mip_tie_resolution",
        "purview_tie_resolution",
        "sia_tie_resolution",
    ):
        strategy = getattr(iit, field_name)
        components = _strategy_components(strategy)
        if not components:
            raise UnregisteredValueError(
                f"formalism.iit.{field_name}={strategy!r} must be a "
                f"tie-resolution strategy name or a non-empty sequence of "
                f"names. Registered strategies: {sorted(valid)}."
            )
        for component in components:
            _require_registered(
                f"formalism.iit.{field_name}",
                component,
                valid,
                "tie-resolution strategy",
                component=True,
            )
    return None


@register_constraint("background_conditioning_compatible_with_version")
def _background_conditioning_compatible_with_version(config: Any) -> str | None:
    """The cause-side background-conditioning convention must be one the
    active formalism defines.

    PyPhi's IIT 3.0 formalism fixes background units at their observed
    current state on the cause side (the PyPhi 1.x / post-2014-literature
    convention, pinned by the shipped preset). Oizumi et al. (2014) itself
    fixed them at their actual past state for causes (IIT 4.0, S2 Text).
    Pairing it with IIT 4.0's causal marginalization silently
    computes a different phi on proper-subset systems. Formalisms that accept
    any registered convention declare
    ``compatible_background_conditioning = None`` and are not constrained.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return None
    compatible = getattr(formalism, "compatible_background_conditioning", None)
    if compatible is None:
        return None  # unconstrained (e.g. IIT 4.0)
    value = iit.background_conditioning
    if value not in compatible:
        return (
            f"formalism.iit.background_conditioning={value!r} is not "
            f"compatible with formalism.iit.version={version!r}. Compatible "
            f"conventions for this version: {sorted(compatible)}. Fix: set "
            f"formalism.iit.background_conditioning to one of those (the "
            f"shipped IIT 3.0 preset uses 'CONDITION_CURRENT_STATE'), or "
            f"change formalism.iit.version."
        )
    return None


@register_constraint("system_measure_matches_requirement")
def _system_measure_matches_requirement(config: Any) -> str | None:
    """The system measure applies the intrinsic-information requirement
    (Eq. 23) exactly when the version is defined by it.

    A mismatch computes one version's quantity while reporting another's; see
    the module Notes. Registered after the measure constraint, so
    ``system_phi_measure`` is known resolvable here.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return None
    if not getattr(formalism, "uses_system_phi_measure", False):
        return None
    from pyphi.measures.distribution import resolve_system_measure

    required = getattr(formalism, "applies_intrinsic_information_requirement", False)
    applied = getattr(
        resolve_system_measure(iit.system_phi_measure),
        "applies_intrinsic_information_requirement",
        False,
    )
    if required == applied:
        return None
    does = "does" if applied else "does not"
    fix = "'INTRINSIC_INFORMATION'" if required else "'GENERALIZED_INTRINSIC_DIFFERENCE'"
    return (
        f"formalism.iit.system_phi_measure={iit.system_phi_measure!r} {does} "
        f"apply the intrinsic-information requirement (Eq. 23), which "
        f"formalism.iit.version={version!r} "
        f"{'requires' if required else 'excludes'}. Fix: select the version "
        f"with the formalism= argument or a preset (pyphi.iit4_2023, "
        f"pyphi.iit4_2026), which set every field together, or set "
        f"formalism.iit.system_phi_measure to {fix}."
    )


@register_constraint("actual_causation_formalism_registered")
def _actual_causation_formalism_registered(config: Any) -> str | None:
    """The actual-causation version must be registered, and its alpha measure
    one that version accepts."""
    try:
        from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY
    except ImportError:
        return None  # bootstrap window; see _FORMALISM_UNAVAILABLE
    registry = ACTUAL_CAUSATION_FORMALISM_REGISTRY
    if not len(registry):
        return None  # the formalisms are still being registered
    ac = config.formalism.actual_causation
    _require_registered(
        "formalism.actual_causation.version",
        ac.version,
        set(registry),
        "actual-causation formalism",
    )
    _require_registered(
        "formalism.actual_causation.alpha_measure",
        ac.alpha_measure,
        registry[ac.version].compatible_measures,
        "alpha measures",
        accepted_by=ac.version,
    )
    return None
