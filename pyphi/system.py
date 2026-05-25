"""System value type — ``(Substrate, state, node_subset, partition)``.

A System is the unit of analysis for IIT: a substrate evaluated in a specific
state over a specific subset of its nodes, optionally with a partition
applied. Immutable, hashable. The partition is a constructor argument
(default :class:`NullCut`, i.e. no edges severed), not a hidden mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

from pyphi import connectivity
from pyphi import utils
from pyphi import validate
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import NullCut
from pyphi.substrate import Substrate

from .core.tpm.factored import FactoredTPM
from .core.tpm.marginalization import _cause_tpm_factored_kary
from .core.tpm.marginalization import cause_tpm as _marginalize_cause
from .core.tpm.marginalization import effect_tpm as _marginalize_effect

if TYPE_CHECKING:
    from pyphi.types import NodeIndices
    from pyphi.types import State


@dataclass(frozen=True, eq=False)
class System:
    """A substrate evaluated in a specific state over a node subset, with partition."""

    substrate: Substrate
    state: State
    node_indices: NodeIndices = field(default=None)  # type: ignore[assignment]
    partition: DirectedBipartition = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        substrate = self.substrate
        validate.state_length(self.state, substrate.size)
        validate.node_states(self.state)
        if self.node_indices is None:
            object.__setattr__(self, "node_indices", tuple(range(substrate.size)))
        else:
            object.__setattr__(
                self,
                "node_indices",
                substrate.node_labels.coerce_to_indices(self.node_indices),
            )
        if self.partition is None:
            object.__setattr__(
                self, "partition", NullCut(self.node_indices, substrate.node_labels)
            )
        else:
            cut_idx = getattr(self.partition, "indices", None)
            if cut_idx is not None and set(cut_idx) != set(self.node_indices):
                raise ValueError(
                    f"{self.partition} nodes are not equal to "
                    f"system nodes {self.node_indices}"
                )
        from pyphi.conf import config as _config

        if _config.infrastructure.validate_system_states:
            validate.state_reachable(self)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) < len(other.node_indices)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) <= len(other.node_indices)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) > len(other.node_indices)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) >= len(other.node_indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.state == other.state
            and self.node_indices == other.node_indices
            and self.partition == other.partition
        )

    def __hash__(self) -> int:
        return hash((self.substrate, self.state, self.node_indices, self.partition))

    def __len__(self) -> int:
        return len(self.node_indices)

    def __str__(self) -> str:
        labels = self.node_labels.coerce_to_labels(self.node_indices)
        return f"System({', '.join(str(label) for label in labels)})"

    def apply_cut(self, partition: DirectedBipartition) -> System:
        """Return a new System with the given partition applied.

        ``substrate``, ``state``, and ``node_indices`` are unchanged.
        Cached derived properties that don't depend on the partition (``cause_tpm``,
        ``effect_tpm``) are not re-derived in the new instance until first
        access, so the new instance shares the same numerical values.
        """
        from dataclasses import replace

        return replace(self, partition=partition)

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        state: Any,
        nodes: Any | None = None,
        partition: DirectedBipartition | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> System:
        """Construct a System from a substrate, state, and optional node subset."""
        if nodes is None:
            nodes = tuple(range(substrate.size))
        return cls(
            substrate=substrate,
            state=tuple(state),
            node_indices=tuple(nodes),
            partition=partition,  # type: ignore[arg-type]
        )

    # ---- cached cheap derived properties ----

    @cached_property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        all_indices = set(range(self.substrate.size))
        return tuple(sorted(all_indices - set(self.node_indices)))

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def _typed_tpm(self) -> Any:
        """The canonical FactoredTPM stored on the substrate."""
        return self.substrate.factored_tpm

    @cached_property
    def cause_tpm(self) -> FactoredTPM:
        """Per-output-unit cause factors for the system; see IIT 4.0 Eq. 4."""
        return _marginalize_cause(
            self._typed_tpm,
            self.state,
            self.node_indices,
        )

    @cached_property
    def effect_tpm(self) -> Any:
        from .core.tpm.factored import FactoredTPM
        from .core.tpm.joint import JointTPM

        external_state = utils.state_of(self.external_indices, self.state)
        background = dict(zip(self.external_indices, external_state, strict=False))
        typed = _marginalize_effect(
            self._typed_tpm,  # type: ignore[arg-type]
            background,
        )
        if isinstance(typed, FactoredTPM) and all(a == 2 for a in typed.alphabet_sizes):
            import numpy as np

            n = typed.n_nodes
            sbn = np.stack([typed.factor(i)[..., 1] for i in range(n)], axis=-1)
            return JointTPM(sbn)._inner
        return typed._inner if hasattr(typed, "_inner") else typed  # type: ignore[union-attr]

    @cached_property
    def proper_effect_tpm(self) -> Any:
        import numpy as np

        return np.asarray(self.effect_tpm.squeeze())[..., list(self.node_indices)]

    @cached_property
    def proper_cause_tpm(self) -> FactoredTPM:
        """Cause TPM restricted to system units.

        Per system unit ``i`` in ``node_indices``, the returned FactoredTPM
        carries the cause factor produced by Bayesian inversion of the
        substrate's forward TPM under the observed state. Background units
        are marginalized via ``pr_bg / norm`` weighting per IIT 4.0 Eq. 4
        and dropped from each factor's input dims, so the returned shape
        is ``(*system_alphabet, k_i)`` per system output unit.
        """
        import numpy as np

        factored = _cause_tpm_factored_kary(
            self._typed_tpm,
            self.state,
            self.node_indices,
        )
        background_indices = tuple(
            i for i in range(factored.n_nodes) if i not in set(self.node_indices)
        )
        system_factors = []
        for i in self.node_indices:
            f = factored.factor(i)
            if background_indices:
                f = np.squeeze(f, axis=background_indices)
            system_factors.append(f)
        return FactoredTPM(factors=system_factors)

    @cached_property
    def cm(self) -> Any:
        return self.partition.apply_cut(self.substrate.cm)

    @cached_property
    def proper_cm(self) -> Any:
        return connectivity.subadjacency(self.cm, self.node_indices)

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def partition_indices(self) -> NodeIndices:
        return self.node_indices

    @cached_property
    def partition_node_labels(self) -> Any:
        from pyphi.labels import NodeLabels

        if self.partition_indices == self.node_indices:
            return self.node_labels
        labels = self.node_labels.coerce_to_labels(self.partition_indices)
        return NodeLabels(labels, self.partition_indices)

    @cached_property
    def is_partitioned(self) -> bool:
        return not isinstance(self.partition, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.substrate.size

    @cached_property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        return generate_nodes(
            self.cause_tpm,
            self.effect_tpm._inner
            if hasattr(self.effect_tpm, "_inner")
            else self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def partitioned_mechanisms(self) -> Any:
        return list(self.partition.all_cut_mechanisms())

    @cached_property
    def _index2node(self) -> dict[int, Any]:
        return {node.index: node for node in self.nodes}

    @cached_property
    def null_distinction(self) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.null_distinction(self)

    @cached_property
    def null_concept(self) -> Any:
        """IIT 3.0 alias for :attr:`null_distinction`."""
        return self.null_distinction

    # ---- repertoire algebra proxies ----

    def cause_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_repertoire(self, mechanism, purview, **kwargs)

    def effect_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_repertoire(self, mechanism, purview, **kwargs)

    def repertoire(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.repertoire(self, direction, mechanism, purview, **kwargs)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction: Any, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(
        self,
        direction: Any,
        partition: Any,
        *,
        mechanism_measure: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.partitioned_repertoire(
            self, direction, partition, mechanism_measure=mechanism_measure, **kwargs
        )

    def expand_repertoire(
        self,
        direction: Any,
        repertoire_array: Any,
        new_purview: Any | None = None,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_repertoire(
            self, direction, repertoire_array, new_purview=new_purview
        )

    def expand_cause_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_cause_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def expand_effect_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_effect_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def forward_cause_repertoire(
        self, mechanism: Any, purview: Any, purview_state: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(
        self, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_repertoire(self, mechanism, purview, **kwargs)

    def forward_repertoire(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_repertoire(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(self, mechanism, purview)

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(self, mechanism, purview)

    def unconstrained_forward_repertoire(
        self, direction: Any, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(self, direction, mechanism, purview)

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kwargs: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_probability(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    # ---- info / phi proxies ----

    def cause_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(_config.formalism.iit.mechanism_phi_measure),
        )
        return ra.cause_info(self, mechanism, purview, **kwargs)

    def effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(_config.formalism.iit.mechanism_phi_measure),
        )
        return ra.effect_info(self, mechanism, purview, **kwargs)

    def cause_effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(_config.formalism.iit.mechanism_phi_measure),
        )
        return ra.cause_effect_info(self, mechanism, purview, **kwargs)

    def intrinsic_information(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        *,
        specification_measure: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.intrinsic_information(
            self,
            direction,
            mechanism,
            purview,
            specification_measure=specification_measure,
            **kwargs,
        )

    # ---- formalism queries ----
    #
    # System-level entry points (``sia``, ``ces``, ``distinctions``) and the
    # mechanism-level queries used to build them (``find_mip``, ``find_mice``,
    # ``distinction``, ``all_distinctions``, ``evaluate_partition``, …) are
    # exposed as thin convenience methods that dispatch via the active
    # formalism. The same operations live as free functions in
    # :mod:`pyphi.formalism` for callers who prefer that grammar.

    def sia(self, **kwargs: Any) -> Any:
        """Return the system irreducibility analysis under the active formalism.

        Resolves the system- and specification-level measures from config at
        the public boundary and threads them to the active formalism
        explicitly, so formalism methods are never called without explicit
        measures in normal flow.
        """
        from pyphi.conf import config as _config
        from pyphi.formalism import sia as _sia
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.measures.distribution import resolve_system_measure

        if _config.formalism.iit.version != "IIT_3_0":
            kwargs.setdefault(
                "system_measure",
                resolve_system_measure(_config.formalism.iit.system_phi_measure),
            )
            kwargs.setdefault(
                "specification_measure",
                resolve_mechanism_measure(_config.formalism.iit.specification_measure),
            )
        return _sia(self, **kwargs)

    def ces(self, **kwargs: Any) -> Any:
        """Return the cause-effect structure of this system (Eq. 57).

        Under IIT 4.0 returns a :class:`CauseEffectStructure` (distinctions
        plus their relations). Under IIT 3.0 returns the
        :class:`Distinctions` (IIT 3.0 has no relations, so the CES is
        exactly the set of distinctions).
        """
        from pyphi.conf import config as _config

        formalism_name = _config.formalism.iit.version
        if formalism_name == "IIT_3_0":
            from pyphi.formalism.iit3 import (
                _compute_distinctions as _ces,  # pyright: ignore[reportPrivateUsage]
            )

            return _ces(self, **kwargs)
        from pyphi.formalism.iit4 import ces as _ces
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.measures.distribution import resolve_system_measure

        kwargs.setdefault(
            "system_measure",
            resolve_system_measure(_config.formalism.iit.system_phi_measure),
        )
        kwargs.setdefault(
            "specification_measure",
            resolve_mechanism_measure(_config.formalism.iit.specification_measure),
        )
        return _ces(self, **kwargs)

    def distinctions(self, **kwargs: Any) -> Any:
        """Return the :class:`Distinctions` of this system.

        The set of irreducible cause-effect distinctions specified by
        mechanisms in the system, without the relations that bind them.
        """
        from pyphi.conf import config as _config

        formalism_name = _config.formalism.iit.version
        if formalism_name == "IIT_3_0":
            from pyphi.formalism.iit3 import (
                _compute_distinctions as _distinctions,  # pyright: ignore[reportPrivateUsage]
            )

            return _distinctions(self, **kwargs)
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self, **kwargs)

    def find_mip(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        """Return the minimum information partition for a mechanism over a purview.

        Resolves mechanism- and specification-level measures from config at
        the public boundary so the active formalism's MIP search is never
        called without explicit measures in normal flow. Explicit
        ``mechanism_measure``/``specification_measure`` kwargs override.
        """
        from pyphi.conf import config as _config
        from pyphi.formalism import find_mip as _find_mip
        from pyphi.measures.distribution import resolve_mechanism_measure

        if _config.formalism.iit.version != "IIT_3_0":
            kwargs.setdefault(
                "mechanism_measure",
                resolve_mechanism_measure(_config.formalism.iit.mechanism_phi_measure),
            )
            kwargs.setdefault(
                "specification_measure",
                resolve_mechanism_measure(_config.formalism.iit.specification_measure),
            )
        return _find_mip(self, direction, mechanism, purview, **kwargs)

    def cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import cause_mip as _cause_mip

        return _cause_mip(self, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import effect_mip as _effect_mip

        return _effect_mip(self, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_cause_mip as _phi_cause_mip

        return _phi_cause_mip(self, mechanism, purview, **kwargs)

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_effect_mip as _phi_effect_mip

        return _phi_effect_mip(self, mechanism, purview, **kwargs)

    def phi(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi as _phi

        return _phi(self, mechanism, purview, **kwargs)

    def find_mice(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import find_mice as _find_mice

        return _find_mice(self, direction, mechanism, **kwargs)

    def mic(self, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import mic as _mic

        return _mic(self, mechanism, **kwargs)

    def mie(self, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import mie as _mie

        return _mie(self, mechanism, **kwargs)

    def phi_max(self, mechanism: Any) -> float:
        from pyphi.formalism import phi_max as _phi_max

        return _phi_max(self, mechanism)

    def distinction(self, mechanism: Any) -> Any:
        from pyphi.formalism import distinction as _distinction

        return _distinction(self, mechanism)

    def all_distinctions(self, **kwargs: Any) -> Any:
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self, **kwargs)

    def evaluate_partition(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.formalism import evaluate_partition as _evaluate_partition

        return _evaluate_partition(
            self, direction, mechanism, purview, partition, **kwargs
        )

    def potential_purviews(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.potential_purviews(self, direction, mechanism, **kwargs)

    def indices2nodes(self, indices: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.indices2nodes(self, indices)

    # ---- cache surface + serialization ----

    def cache_info(self) -> dict[str, Any]:
        from pyphi.core import repertoire_algebra as ra

        return ra.cache_info()

    def clear_caches(self) -> None:
        from pyphi.core import repertoire_algebra as ra

        ra.clear_caches(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "state": list(self.state),
            "node_indices": list(self.node_indices),
            "partition": self.partition,
        }
