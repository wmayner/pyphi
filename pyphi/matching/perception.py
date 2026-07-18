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

    A view over a cause-effect structure that computes how much of the
    structure's cause-effect power was triggered by ``stimulus``, without
    modifying the structure. ``ces`` must be the structure triggered by
    ``stimulus``: its system state must equal the state the stimulus triggers,
    which ``__post_init__`` checks against ``triggered_tpm``.

    Attributes
    ----------
    ces : CauseEffectStructure
        The Φ-structure unfolded from the triggered system state.
    triggered_tpm : TriggeredTPM
        The fixed-lag response distribution supplying triggering coefficients.
    stimulus : tuple of int
        The sensory-interface state that triggered ``ces``.
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
        """Mapping ``{mechanism: TriggeringCoefficient}``, one per distinction.

        Keyed by each distinction's mechanism, evaluated at that distinction's
        mechanism state for this stimulus.
        """
        return {
            d.mechanism: triggering_coefficient(
                self.triggered_tpm, d.mechanism, d.mechanism_state, self.stimulus
            )
            for d in self.ces.distinctions
        }

    def distinction_perception(self, distinction) -> float:
        """Perception value of a distinction, t(x, m) · φ_d (Eq. 8)."""
        t = self.triggering_coefficients[distinction.mechanism].value
        return t * float(distinction.phi)

    def relation_perception(self, relation) -> float:
        """Perception value of a relation, t(x, r(d)) · φ_r (Eqs. 9-10).

        The relation's triggering coefficient t(x, r(d)) is the unweighted mean
        of the triggering coefficients of the distinctions it binds (Eq. 9), so
        the perception value is the full relation φ_r times that mean (Eq. 10).
        """
        mean_t = float(
            np.mean(
                [self.triggering_coefficients[rel.mechanism].value for rel in relation]
            )
        )
        return float(relation.phi) * mean_t

    def fold_perception(self, fold: PhiFold) -> float:
        """Perception value of a single distinction's Φ-fold (Eq. 11).

        For the Φ-fold of a mechanism m — the distinction and every relation
        involving it — returns the seed's triggering coefficient t(x, m)
        times the fold's contribution to Φ (Eq. 3): the distinction's φ plus
        each incident relation's φ divided by its degree. ``fold`` must
        contain exactly one distinction (its seed).
        """
        (seed,) = fold.distinctions
        t = self.triggering_coefficients[seed.mechanism].value
        return t * fold.big_phi_contribution

    @cached_property
    def richness(self) -> float:
        """Total perceptual richness P(x, y), summed over the structure (Eq. 13).

        The sum of the perception values of every distinction and relation in
        ``ces``: the quantity of intrinsic meaning the stimulus triggered.
        """
        distinctions = sum(self.distinction_perception(d) for d in self.ces.distinctions)
        relations = sum(
            self.relation_perception(r)
            for r in self.ces.relations  # pyright: ignore[reportGeneralTypeIssues]  # Relations base lacks __iter__; concrete subclasses provide it
        )
        return distinctions + relations
