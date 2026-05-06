"""CandidateSystem value type — (CausalModel, state, node_subset, cut).

The replacement for :class:`pyphi.subsystem.Subsystem`. Immutable.
Hashable. Cut is a constructor argument, not a hidden mode.

P7 stages:
- Task 3.1: skeleton: construction validation, equality, hash,
  default-NullCut field.
- Task 3.2: cached derived properties (cause_tpm, effect_tpm, cm, etc.).
- Task 3.3: ``apply_cut``.
- Task 5.1: proxy methods to ``repertoire_algebra`` and the formalism.
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
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition

from .causal_model import CausalModel
from .tpm.marginalization import cause_tpm as _marginalize_cause
from .tpm.marginalization import effect_tpm as _marginalize_effect

if TYPE_CHECKING:
    from pyphi.types import NodeIndices
    from pyphi.types import State


@dataclass(frozen=True, eq=False)
class CandidateSystem:
    """A candidate system: ``(CausalModel, state, node_subset, cut)``."""

    causal_model: CausalModel
    state: State
    node_indices: NodeIndices
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        substrate = self.causal_model.substrate
        validate.state_length(self.state, substrate.n_units)
        validate.node_states(self.state)
        # Coerce node_indices to plain Python ints, dedup, sort — matches
        # legacy ``NodeLabels.coerce_to_indices``.
        object.__setattr__(
            self,
            "node_indices",
            tuple(sorted({int(i) for i in self.node_indices})),
        )
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, substrate.node_labels)
            )
        from pyphi.conf import config as _config

        if _config.VALIDATE_SUBSYSTEM_STATES:
            validate.state_reachable(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateSystem):
            return NotImplemented
        return (
            self.causal_model == other.causal_model
            and self.state == other.state
            and self.node_indices == other.node_indices
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash((self.causal_model, self.state, self.node_indices, self.cut))

    def __len__(self) -> int:
        return len(self.node_indices)

    def apply_cut(self, cut: SystemPartition) -> CandidateSystem:
        """Return a new CandidateSystem with the given cut applied.

        ``causal_model``, ``state``, and ``node_indices`` are unchanged.
        Cached derived properties that don't depend on cut (``cause_tpm``,
        ``effect_tpm``) are not re-derived in the new instance until first
        access — so the new instance shares the same numerical values.
        """
        from dataclasses import replace

        return replace(self, cut=cut)

    @classmethod
    def from_network(
        cls,
        network: Any,
        state: Any,
        nodes: Any | None = None,
        cut: SystemPartition | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> CandidateSystem:
        """Build a CandidateSystem from a legacy Network.

        Migration helper that mirrors ``Subsystem(network, state, nodes, cut=...)``
        construction.
        """
        causal_model = CausalModel.from_network(network)
        if nodes is None:
            nodes = tuple(range(network.size))
        return cls(
            causal_model=causal_model,
            state=tuple(state),
            node_indices=tuple(nodes),
            cut=cut,  # type: ignore[arg-type]
        )

    # ---- cached cheap derived properties ----

    @cached_property
    def network(self) -> Any:
        """Back-compat: a legacy Network instance for code paths that still
        require it (during the worktree, deleted at 2.0).
        """
        from pyphi.network import Network

        return Network(
            self.causal_model.tpm.to_array(),
            cm=self.causal_model.substrate.connectivity_matrix,
            node_labels=tuple(self.causal_model.substrate.node_labels),
        )

    @cached_property
    def node_labels(self) -> Any:
        return self.causal_model.substrate.node_labels

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        all_indices = set(range(self.causal_model.substrate.n_units))
        return tuple(sorted(all_indices - set(self.node_indices)))

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def cause_tpm(self) -> Any:
        return _marginalize_cause(
            self.causal_model.tpm,  # type: ignore[arg-type]
            self.state,
            self.node_indices,
        )

    @cached_property
    def effect_tpm(self) -> Any:
        external_state = utils.state_of(self.external_indices, self.state)
        background = dict(zip(self.external_indices, external_state, strict=False))
        return _marginalize_effect(
            self.causal_model.tpm,  # type: ignore[arg-type]
            background,
        )

    @cached_property
    def proper_effect_tpm(self) -> Any:
        return self.effect_tpm.squeeze().to_array()[..., list(self.node_indices)]

    @cached_property
    def proper_cause_tpm(self) -> Any:
        return self.cause_tpm.squeeze().to_array()[..., list(self.node_indices)]

    @cached_property
    def cm(self) -> Any:
        return self.cut.apply_cut(self.causal_model.substrate.connectivity_matrix)

    @cached_property
    def proper_cm(self) -> Any:
        return connectivity.subadjacency(self.cm, self.node_indices)

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def cut_indices(self) -> NodeIndices:
        return self.node_indices

    @cached_property
    def cut_node_labels(self) -> Any:
        return self.node_labels.coerce_to_labels(self.cut_indices)

    @cached_property
    def is_cut(self) -> bool:
        return not isinstance(self.cut, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.causal_model.substrate.n_units

    @cached_property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        return generate_nodes(
            self.cause_tpm._inner
            if hasattr(self.cause_tpm, "_inner")
            else self.cause_tpm,
            self.effect_tpm._inner
            if hasattr(self.effect_tpm, "_inner")
            else self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def cut_mechanisms(self) -> Any:
        return list(self.cut.all_cut_mechanisms())

    @cached_property
    def _index2node(self) -> dict[int, Any]:
        return {node.index: node for node in self.nodes}

    @cached_property
    def null_concept(self) -> Any:
        from . import repertoire_algebra as ra

        return ra.null_concept(self)

    # ---- repertoire algebra proxies ----

    def cause_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.cause_repertoire(self, mechanism, purview, **kwargs)

    def effect_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.effect_repertoire(self, mechanism, purview, **kwargs)

    def repertoire(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.repertoire(self, direction, mechanism, purview, **kwargs)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction: Any, purview: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(
        self, direction: Any, partition: Any, **kwargs: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.partitioned_repertoire(self, direction, partition, **kwargs)

    def expand_repertoire(
        self,
        direction: Any,
        repertoire_array: Any,
        *,
        new_purview: Any | None = None,
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.expand_repertoire(
            self, direction, repertoire_array, new_purview=new_purview
        )

    def expand_cause_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.expand_cause_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def expand_effect_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.expand_effect_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def forward_cause_repertoire(
        self, mechanism: Any, purview: Any, purview_state: Any | None = None
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(
        self, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.forward_effect_repertoire(self, mechanism, purview, **kwargs)

    def forward_repertoire(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.forward_repertoire(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(self, mechanism, purview)

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(self, mechanism, purview)

    def unconstrained_forward_repertoire(
        self, direction: Any, mechanism: Any, purview: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(self, direction, mechanism, purview)

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        from . import repertoire_algebra as ra

        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        from . import repertoire_algebra as ra

        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kwargs: Any,
    ) -> float:
        from . import repertoire_algebra as ra

        return ra.forward_probability(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    # ---- info / phi proxies ----

    def cause_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.cause_info(self, mechanism, purview, **kwargs)

    def effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.effect_info(self, mechanism, purview, **kwargs)

    def cause_effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.cause_effect_info(self, mechanism, purview, **kwargs)

    def intrinsic_information(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.intrinsic_information(self, direction, mechanism, purview, **kwargs)

    def evaluate_partition(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.evaluate_partition(
            self, direction, mechanism, purview, partition, **kwargs
        )

    def find_mip(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra.find_mip(self, direction, mechanism, purview, **kwargs)

    def _find_mip_single_state(
        self,
        specified_state: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        repertoire: Any,
        partitions: Any,
        parallel_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        from . import repertoire_algebra as ra

        return ra._find_mip_single_state(
            self,
            specified_state,
            direction,
            mechanism,
            purview,
            repertoire,
            partitions,
            parallel_kwargs,
            **kwargs,
        )

    def cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.cause_mip(self, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.effect_mip(self, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.phi_cause_mip(self, mechanism, purview, **kwargs)

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.phi_effect_mip(self, mechanism, purview, **kwargs)

    def phi(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.phi(self, mechanism, purview, **kwargs)

    def find_mice(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.find_mice(self, direction, mechanism, **kwargs)

    def mic(self, mechanism: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.mic(self, mechanism, **kwargs)

    def mie(self, mechanism: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.mie(self, mechanism, **kwargs)

    def phi_max(self, mechanism: Any) -> float:
        from . import repertoire_algebra as ra

        return ra.phi_max(self, mechanism)

    def concept(self, mechanism: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.concept(self, mechanism, **kwargs)

    def distinction(self, mechanism: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.distinction(self, mechanism)

    def all_distinctions(self, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.all_distinctions(self, **kwargs)

    def sia(self, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.sia(self, **kwargs)

    def potential_purviews(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.potential_purviews(self, direction, mechanism, **kwargs)

    def indices2nodes(self, indices: Any) -> Any:
        from . import repertoire_algebra as ra

        return ra.indices2nodes(self, indices)

    # ---- cache surface + serialization ----

    def cache_info(self) -> dict[str, Any]:
        from . import repertoire_algebra as ra

        return ra.cache_info()

    def clear_caches(self) -> None:
        from . import repertoire_algebra as ra

        ra.clear_caches(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "causal_model": self.causal_model,
            "state": list(self.state),
            "node_indices": list(self.node_indices),
            "cut": self.cut,
        }
