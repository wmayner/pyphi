# pyright: strict
"""Runtime-checkable Protocols for PyPhi's dispatch points and core abstractions.

Defines the structural contracts that registered metrics, partition schemes,
and phi formalisms must satisfy. The corresponding registries
(``pyphi.metrics.distribution.measures``, ``pyphi.partition.partition_types``)
validate registered objects against these Protocols at registration time, so
wrong-shape registrations fail at import — not at the bottom of a long phi
computation.

Also declares the public-surface contract for ``Subsystem`` as
:class:`SubsystemPublicInterface`. This is the cross-module contract the
forthcoming subsystem rewrite must satisfy. ``test/test_subsystem_surface.py``
fails CI if ``Subsystem``'s public surface drifts from this declaration.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Protocol
from typing import runtime_checkable

__all__ = [
    "PUBLIC_SUBSYSTEM_ATTRS",
    "DistanceMetric",
    "PartitionScheme",
    "PhiFormalism",
    "SubsystemInternalInterface",
    "SubsystemPublicInterface",
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
# Partition schemes
# =============================================================================
@runtime_checkable
class PartitionScheme(Protocol):
    """Callable that yields the partitions of a (mechanism, purview) pair.

    Concrete schemes registered in ``pyphi.partition.partition_types``
    (``BI``, ``TRI``, ``ALL``, etc.) all share this signature. Each yields
    an iterable of partition objects — :class:`pyphi.models.cuts.Bipartition`,
    :class:`pyphi.models.cuts.Tripartition`, etc. — that the MIP search
    enumerates.
    """

    def __call__(
        self,
        mechanism: Any,
        purview: Any,
        node_labels: Any = None,
    ) -> Iterable[Any]: ...


# =============================================================================
# Phi formalisms (placeholder — full shape lands with the formalism split)
# =============================================================================
class PhiFormalism(Protocol):
    """Top-level strategy for computing integrated information.

    A formalism bundles a partition scheme, a compatible distance metric, and
    the algorithms that combine them into mechanism-level RIAs, system-level
    SIAs, and Φ-structures. The full shape is defined when IIT 3.0 and IIT
    4.0 are separated into ``pyphi/formalism/iit3/`` and
    ``pyphi/formalism/iit4/``; this declaration captures only the method
    signatures the formalism split target requires, so that downstream code
    can begin annotating against it.

    Approximate methods (φ\\*, φ_G, etc.) will subtype this Protocol with an
    additional ``error_characterization`` method.
    """

    default_metric: DistanceMetric
    partition_scheme: PartitionScheme | None

    def evaluate_mechanism(
        self, candidate_system: Any, mechanism: Any, purview: Any
    ) -> Any: ...

    def evaluate_system(self, candidate_system: Any) -> Any: ...

    def build_phi_structure(self, candidate_system: Any) -> Any: ...


# =============================================================================
# Subsystem public surface
# =============================================================================
PUBLIC_SUBSYSTEM_ATTRS: frozenset[str] = frozenset(
    {
        # Construction-time attributes
        "cause_tpm",
        "cm",
        "cut",
        "effect_tpm",
        "external_indices",
        "network",
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
        # Mechanism / system analysis
        "cause_info",
        "effect_info",
        "cause_effect_info",
        "intrinsic_information",
        "phi",
        "phi_cause_mip",
        "phi_effect_mip",
        "phi_max",
        "cause_mip",
        "effect_mip",
        "find_mip",
        "find_mice",
        "evaluate_partition",
        "mic",
        "mie",
        "concept",
        "distinction",
        "all_distinctions",
        "sia",
        # Utilities
        "apply_cut",
        "potential_purviews",
        "indices2nodes",
        "cache_info",
        "clear_caches",
        "to_json",
    }
)
"""Names that must remain on ``Subsystem``'s public surface.

Maintained in lockstep with the class via ``test/test_subsystem_surface.py``,
which fails CI on any drift. Tighten or expand this set deliberately as part
of an explicit refactor; do not let drift accumulate."""


@runtime_checkable
class SubsystemPublicInterface(Protocol):
    """The cross-module contract for ``Subsystem``.

    Generated from ``dir(Subsystem)`` plus instance attributes set in
    ``__init__``, filtered to the names that actually appear in cross-module
    accesses inside ``pyphi/``. Internal-only members (callable from inside
    ``Subsystem`` itself but not by external callers) are kept on the class
    but excluded from this Protocol.

    The members are typed ``Any`` here because their concrete signatures are
    in flux until the formalism split and metric-API unification land. The
    Protocol's role at this stage is **structural conformance and drift
    detection** — the type system enforces that any caller annotated against
    ``SubsystemPublicInterface`` only touches names that ``Subsystem``
    actually exposes. Concrete signatures will be added incrementally.
    """

    # Construction-time attributes
    cause_tpm: Any
    cm: Any
    cut: Any
    effect_tpm: Any
    external_indices: Any
    network: Any
    node_indices: Any
    node_labels: Any
    nodes: Any
    proper_cause_tpm: Any
    proper_cm: Any
    proper_effect_tpm: Any
    state: Any
    # Properties
    connectivity_matrix: Any
    cut_indices: Any
    cut_mechanisms: Any
    cut_node_labels: Any
    is_cut: Any
    null_concept: Any
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
    # Mechanism / system analysis
    cause_info: Callable[..., Any]
    effect_info: Callable[..., Any]
    cause_effect_info: Callable[..., Any]
    intrinsic_information: Callable[..., Any]
    phi: Callable[..., Any]
    phi_cause_mip: Callable[..., Any]
    phi_effect_mip: Callable[..., Any]
    phi_max: Callable[..., Any]
    cause_mip: Callable[..., Any]
    effect_mip: Callable[..., Any]
    find_mip: Callable[..., Any]
    find_mice: Callable[..., Any]
    evaluate_partition: Callable[..., Any]
    mic: Callable[..., Any]
    mie: Callable[..., Any]
    concept: Callable[..., Any]
    distinction: Callable[..., Any]
    all_distinctions: Callable[..., Any]
    sia: Callable[..., Any]
    # Utilities
    apply_cut: Callable[..., Any]
    potential_purviews: Callable[..., Any]
    indices2nodes: Callable[..., Any]
    cache_info: Callable[..., Any]
    clear_caches: Callable[..., Any]
    to_json: Callable[..., Any]


class SubsystemInternalInterface(Protocol):
    """Internal-only members of ``Subsystem`` — not part of the cross-module
    contract.

    These names are referenced only from ``Subsystem`` itself (and the
    macro/actual-causation modules that subclass it). The forthcoming
    subsystem rewrite is free to rename, restructure, or remove them
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
