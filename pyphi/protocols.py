# pyright: strict
"""Runtime-checkable Protocols for PyPhi's dispatch points and core abstractions.

Defines the structural contracts that registered metrics, partition schemes,
and phi formalisms must satisfy. The corresponding registries
(the typed metric registries in :mod:`pyphi.metrics.distribution`,
``pyphi.partition.partition_types``) validate registered objects against
these Protocols at registration time, so wrong-shape registrations fail
at import — not at the bottom of a long phi computation.

Also declares the public-surface contract for ``System`` as
:class:`SystemPublicInterface`. This is the cross-module contract the
forthcoming system rewrite must satisfy. ``test/test_system_surface.py``
fails CI if ``System``'s public surface drifts from this declaration.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Protocol
from typing import runtime_checkable

__all__ = [
    "PUBLIC_SYSTEM_ATTRS",
    "DistanceMetric",
    "MechanismPartition",
    "MechanismPartitionScheme",
    "SystemInternalInterface",
    "SystemPartitionLike",
    "SystemPartitionScheme",
    "SystemPublicInterface",
]


# =============================================================================
# Distance metrics
# =============================================================================
@runtime_checkable
class DistanceMetric(Protocol):
    """Callable that computes a distance between probability distributions.

    The signature varies between metric families:

    - **Distribution metrics** (e.g., ``EMD``, ``KLD``, ``L1``): take two
      repertoires and return a scalar — ``f(p, q) -> float``.
    - **State-aware metrics** (e.g., ``GENERALIZED_INTRINSIC_DIFFERENCE``,
      ``INTRINSIC_INFORMATION``): take additional ``state``, ``selectivity``,
      and partitioned-repertoire arguments and return a scalar or
      ``DistanceResult``.

    The minimum common contract is callability with positional repertoires
    and arbitrary keyword arguments. A unified, narrower signature is the
    target of the metric-API refactor (see ``pyphi/metrics/distribution.py``
    around ``repertoire_distance``); until that lands, this Protocol
    documents the minimum shape and the registry validates against it.
    """

    def __call__(self, p: Any, q: Any, *args: Any, **kwargs: Any) -> Any: ...


# =============================================================================
# Partitions
# =============================================================================
@runtime_checkable
class MechanismPartition(Protocol):
    """A partition of a mechanism over a purview.

    Concrete realizations are :class:`pyphi.models.cuts.Bipartition`,
    :class:`pyphi.models.cuts.Tripartition`, and the more general
    :class:`pyphi.models.cuts.KPartition`. The partition exposes the union
    of indices on each side via ``mechanism`` and ``purview`` properties; it
    is iterable over its constituent :class:`pyphi.models.cuts.Part` objects.

    This Protocol distinguishes mechanism-level partitions (Eqs. 5-7,
    used by ``System.find_mip`` for distinctions) from system-level
    partitions (Eqs. 14-18, used by SIA). The two have different
    mathematical roles and different probability constructions; making
    the distinction explicit in the type system prevents accidental
    cross-use.
    """

    mechanism: Any
    purview: Any

    def __iter__(self) -> Any: ...

    def __len__(self) -> int: ...


@runtime_checkable
class SystemPartitionLike(Protocol):
    """A directional partition of a set of system nodes.

    Concrete realizations are :class:`pyphi.models.cuts.SystemPartition`
    (the canonical 2.0 type, with explicit ``Direction``) and the general
    set-partition variants (:class:`pyphi.models.cuts.GeneralKCut`,
    :class:`pyphi.models.cuts.GeneralSetPartition`).

    All system-level partitions expose ``cut_matrix(n)`` for applying the
    cut to a connectivity matrix and ``indices`` for the partitioned nodes.
    """

    def cut_matrix(self, n: int) -> Any: ...

    @property
    def indices(self) -> tuple[int, ...]: ...


# =============================================================================
# Partition schemes
# =============================================================================
@runtime_checkable
class MechanismPartitionScheme(Protocol):
    """Callable that yields mechanism-level partitions of a (mechanism, purview) pair.

    Concrete schemes registered in ``pyphi.partition.partition_types``
    (``BI``, ``TRI``, ``ALL``, etc.) all share this signature. Each yields
    an iterable of :class:`MechanismPartition` instances — concrete types
    are :class:`pyphi.models.cuts.Bipartition`,
    :class:`pyphi.models.cuts.Tripartition`, and
    :class:`pyphi.models.cuts.KPartition` — that the MIP search enumerates.

    Distinct from :class:`SystemPartitionScheme` because mechanism-level
    partitions are over a (mechanism, purview) pair rather than a single
    set of nodes.
    """

    def __call__(
        self,
        mechanism: Any,
        purview: Any,
        node_labels: Any = None,
    ) -> Iterable[MechanismPartition]: ...


@runtime_checkable
class SystemPartitionScheme(Protocol):
    """Callable that yields the system-level partitions of a set of nodes.

    Concrete schemes registered in
    ``pyphi.partition.system_partition_types``
    (``DIRECTED_BI``, ``SET_UNI/BI``, ``GENERAL``, etc.). Yields
    :class:`SystemPartitionLike` instances.

    Distinct from :class:`MechanismPartitionScheme` because system-level
    partitions are over a single set of nodes rather than a (mechanism,
    purview) pair, and include a ``Direction`` per partition.
    """

    def __call__(
        self,
        nodes: Any,
        node_labels: Any = None,
    ) -> Iterable[SystemPartitionLike]: ...


# =============================================================================
# System public surface
# =============================================================================
PUBLIC_SYSTEM_ATTRS: frozenset[str] = frozenset(
    {
        # Construction-time attributes
        "cause_tpm",
        "cm",
        "cut",
        "effect_tpm",
        "external_indices",
        "from_substrate",
        "substrate",
        "node_indices",
        "node_labels",
        "nodes",
        "proper_cause_tpm",
        "proper_cm",
        "proper_effect_tpm",
        "state",
        # Properties
        "connectivity_matrix",
        "cut_indices",
        "cut_mechanisms",
        "cut_node_labels",
        "is_cut",
        "null_concept",
        "null_distinction",
        "proper_state",
        "size",
        "tpm_size",
        # Repertoire computation
        "cause_repertoire",
        "effect_repertoire",
        "repertoire",
        "partitioned_repertoire",
        "unconstrained_cause_repertoire",
        "unconstrained_effect_repertoire",
        "unconstrained_repertoire",
        "expand_cause_repertoire",
        "expand_effect_repertoire",
        "expand_repertoire",
        "forward_cause_repertoire",
        "forward_effect_repertoire",
        "forward_repertoire",
        "forward_cause_probability",
        "forward_effect_probability",
        "forward_probability",
        "unconstrained_forward_cause_repertoire",
        "unconstrained_forward_effect_repertoire",
        "unconstrained_forward_repertoire",
        # Mechanism-level info (kernel)
        "cause_info",
        "effect_info",
        "cause_effect_info",
        "intrinsic_information",
        # Formalism-level convenience dispatchers
        "sia",
        "phi_structure",
        "ces",
        "find_mip",
        "cause_mip",
        "effect_mip",
        "phi_cause_mip",
        "phi_effect_mip",
        "phi",
        "find_mice",
        "mic",
        "mie",
        "phi_max",
        "distinction",
        "all_distinctions",
        "evaluate_partition",
        # Utilities
        "apply_cut",
        "potential_purviews",
        "indices2nodes",
        "cache_info",
        "clear_caches",
        "to_json",
    }
)
"""Names that must remain on ``System``'s public surface.

Maintained in lockstep with the class via ``test/test_system_surface.py``,
which fails CI on any drift. Tighten or expand this set deliberately as part
of an explicit refactor; do not let drift accumulate."""


@runtime_checkable
class SystemPublicInterface(Protocol):
    """The cross-module contract for ``System``.

    Generated from ``dir(System)`` plus instance attributes set in
    ``__init__``, filtered to the names that actually appear in cross-module
    accesses inside ``pyphi/``. Internal-only members (callable from inside
    ``System`` itself but not by external callers) are kept on the class
    but excluded from this Protocol.

    The members are typed ``Any`` here because their concrete signatures are
    in flux until the formalism split and metric-API unification land. The
    Protocol's role at this stage is **structural conformance and drift
    detection** — the type system enforces that any caller annotated against
    ``SystemPublicInterface`` only touches names that ``System``
    actually exposes. Concrete signatures will be added incrementally.
    """

    # Construction-time attributes
    cause_tpm: Any
    cm: Any
    cut: Any
    effect_tpm: Any
    external_indices: Any
    substrate: Any
    node_indices: Any
    node_labels: Any
    nodes: Any
    proper_cause_tpm: Any
    proper_cm: Any
    proper_effect_tpm: Any
    state: Any
    from_substrate: Callable[..., Any]
    # Properties
    connectivity_matrix: Any
    cut_indices: Any
    cut_mechanisms: Any
    cut_node_labels: Any
    is_cut: Any
    null_concept: Any
    null_distinction: Any
    proper_state: Any
    size: Any
    tpm_size: Any
    # Repertoire computation (callables)
    cause_repertoire: Callable[..., Any]
    effect_repertoire: Callable[..., Any]
    repertoire: Callable[..., Any]
    partitioned_repertoire: Callable[..., Any]
    unconstrained_cause_repertoire: Callable[..., Any]
    unconstrained_effect_repertoire: Callable[..., Any]
    unconstrained_repertoire: Callable[..., Any]
    expand_cause_repertoire: Callable[..., Any]
    expand_effect_repertoire: Callable[..., Any]
    expand_repertoire: Callable[..., Any]
    forward_cause_repertoire: Callable[..., Any]
    forward_effect_repertoire: Callable[..., Any]
    forward_repertoire: Callable[..., Any]
    forward_cause_probability: Callable[..., Any]
    forward_effect_probability: Callable[..., Any]
    forward_probability: Callable[..., Any]
    unconstrained_forward_cause_repertoire: Callable[..., Any]
    unconstrained_forward_effect_repertoire: Callable[..., Any]
    unconstrained_forward_repertoire: Callable[..., Any]
    # Mechanism-level info (kernel)
    cause_info: Callable[..., Any]
    effect_info: Callable[..., Any]
    cause_effect_info: Callable[..., Any]
    intrinsic_information: Callable[..., Any]
    # Formalism-level convenience dispatchers
    sia: Callable[..., Any]
    phi_structure: Callable[..., Any]
    ces: Callable[..., Any]
    find_mip: Callable[..., Any]
    cause_mip: Callable[..., Any]
    effect_mip: Callable[..., Any]
    phi_cause_mip: Callable[..., Any]
    phi_effect_mip: Callable[..., Any]
    phi: Callable[..., Any]
    find_mice: Callable[..., Any]
    mic: Callable[..., Any]
    mie: Callable[..., Any]
    phi_max: Callable[..., Any]
    distinction: Callable[..., Any]
    all_distinctions: Callable[..., Any]
    evaluate_partition: Callable[..., Any]
    # Utilities
    apply_cut: Callable[..., Any]
    potential_purviews: Callable[..., Any]
    indices2nodes: Callable[..., Any]
    cache_info: Callable[..., Any]
    clear_caches: Callable[..., Any]
    to_json: Callable[..., Any]


class SystemInternalInterface(Protocol):
    """Internal-only members of ``System`` — not part of the cross-module
    contract.

    These names are referenced only from ``System`` itself (and the
    macro/actual-causation modules that subclass it). The forthcoming
    system rewrite is free to rename, restructure, or remove them
    without affecting external callers. They are listed here so changes
    can be tracked — additions or removals should be intentional, not
    incidental.
    """

    cause_info: Callable[..., Any]
    cause_mip: Callable[..., Any]
    distinction: Callable[..., Any]
    effect_info: Callable[..., Any]
    effect_mip: Callable[..., Any]
    find_mice: Callable[..., Any]
    find_mip: Callable[..., Any]
    forward_cause_probability: Callable[..., Any]
    forward_effect_probability: Callable[..., Any]
    forward_probability: Callable[..., Any]
    indices2nodes: Callable[..., Any]
    phi_cause_mip: Callable[..., Any]
    phi_effect_mip: Callable[..., Any]
    phi_max: Callable[..., Any]
    potential_purviews: Callable[..., Any]
    proper_cause_tpm: Any
    proper_effect_tpm: Any
    tpm_size: Any
    unconstrained_cause_repertoire: Callable[..., Any]
    unconstrained_effect_repertoire: Callable[..., Any]
    unconstrained_forward_cause_repertoire: Callable[..., Any]
    unconstrained_forward_effect_repertoire: Callable[..., Any]
    unconstrained_forward_repertoire: Callable[..., Any]
    unconstrained_repertoire: Callable[..., Any]
