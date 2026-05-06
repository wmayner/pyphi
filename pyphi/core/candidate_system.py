"""CandidateSystem value type — (CausalModel, state, node_subset, cut).

The replacement for :class:`pyphi.subsystem.Subsystem`. Immutable.
Hashable. Cut is a constructor argument, not a hidden mode.

P7 stages:
- Task 3.1: skeleton (this file): construction validation, equality, hash,
  default-NullCut field.
- Task 3.2: cached derived properties (cause_tpm, effect_tpm, cm, etc.).
- Task 3.3: ``apply_cut``.
- Task 5.1: proxy methods to ``repertoire_algebra`` and the formalism.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from pyphi import validate
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition

from .causal_model import CausalModel

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
