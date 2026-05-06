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
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, substrate.node_labels)
            )

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
        return self.cut.all_cut_mechanisms()

    @cached_property
    def null_concept(self) -> Any:
        # Delegate to legacy Subsystem until repertoire_algebra and the
        # mechanism-eval machinery are fully ported (Phase 8).
        from pyphi.subsystem import Subsystem

        return Subsystem(
            self.network,
            self.state,
            self.node_indices,
            cut=self.cut,
        ).null_concept
