"""Perception: the portion of a cause-effect structure triggered by a stimulus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .triggering import triggering_coefficient

if TYPE_CHECKING:
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.models.ces import PhiFold

    from .triggered_tpm import TriggeredTPM


@dataclass(frozen=True)
class Perception:
    """The triggering coefficients and perception values for one stimulus.

    A pure view over a cause-effect structure: it computes how much of the
    structure's cause-effect power was triggered by ``stimulus``, without
    modifying the structure. ``ces`` must be the structure triggered by
    ``stimulus`` (its system state equals the stimulus's triggered state).
    """

    ces: CauseEffectStructure
    triggered_tpm: TriggeredTPM
    stimulus: tuple[int, ...]

    def __post_init__(self):
        sia = self.ces.sia
        if tuple(sia.node_indices) != tuple(self.triggered_tpm.system_indices):
            raise ValueError(
                "ces system nodes do not match the triggered TPM system units"
            )
        triggered = self.triggered_tpm.argmax_state(self.stimulus)
        if tuple(sia.current_state) != tuple(triggered):
            raise ValueError(
                f"ces system state {tuple(sia.current_state)} is not the state "
                f"triggered by stimulus {self.stimulus} ({tuple(triggered)})"
            )

    @cached_property
    def triggering_coefficients(self) -> dict:
        """Mapping {mechanism: TriggeringCoefficient}, one per distinction."""
        return {
            d.mechanism: triggering_coefficient(
                self.triggered_tpm, d.mechanism, d.mechanism_state, self.stimulus
            )
            for d in self.ces.distinctions
        }

    def distinction_perception(self, distinction) -> float:
        """t(x, m) * phi_d (Eq 8)."""
        t = self.triggering_coefficients[distinction.mechanism].value
        return t * float(distinction.phi)

    def relation_perception(self, relation) -> float:
        """phi_r * mean over relata of t(x, relatum) (Eq 9-10, full phi_r)."""
        mean_t = float(
            np.mean(
                [self.triggering_coefficients[rel.mechanism].value for rel in relation]
            )
        )
        return float(relation.phi) * mean_t

    def fold_perception(self, fold: PhiFold) -> float:
        """t(x, m) * Phi_d (Eq 11), for the single-distinction fold of m."""
        (seed,) = fold.distinctions
        t = self.triggering_coefficients[seed.mechanism].value
        return t * fold.big_phi_contribution

    @cached_property
    def richness(self) -> float:
        """Total perceptual richness (Eq 13)."""
        distinctions = sum(self.distinction_perception(d) for d in self.ces.distinctions)
        relations = sum(
            self.relation_perception(r)
            for r in self.ces.relations  # pyright: ignore[reportGeneralTypeIssues]  # Relations base lacks __iter__; concrete subclasses provide it
        )
        return distinctions + relations
