"""PerceptualSystem: a system embedded in an environment via a sensory interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyphi import utils

from .triggered_tpm import TriggeredTPM
from .triggered_tpm import _validate_binary_substrate
from .triggered_tpm import _validate_sorted_indices
from .triggered_tpm import build_triggered_tpm

if TYPE_CHECKING:
    from pyphi.substrate import Substrate


@dataclass(frozen=True)
class PerceptualSystem:
    """A system S within a substrate U, coupled to its environment E = U∖S
    through a sensory interface ∂S ⊆ E.

    Produces the fixed-lag triggered TPM and the triggered response state for
    each stimulus (a state of the sensory interface ∂S). Only binary
    substrates are currently supported.

    Attributes
    ----------
    substrate : Substrate
        The full substrate U.
    system_indices : tuple of int
        Indices of the units composing the system S.
    sensory_indices : tuple of int
        Indices of the sensory interface ∂S. Must be non-empty and disjoint
        from ``system_indices``; the environment E is the remaining units.
    """

    substrate: Substrate
    system_indices: tuple[int, ...]
    sensory_indices: tuple[int, ...]

    def __post_init__(self):
        node_indices = set(self.substrate.node_indices)
        system = set(self.system_indices)
        sensory = set(self.sensory_indices)
        if not system <= node_indices:
            raise ValueError(f"system_indices {self.system_indices} not in substrate")
        if not sensory <= node_indices:
            raise ValueError(f"sensory_indices {self.sensory_indices} not in substrate")
        if not system or not sensory:
            raise ValueError("system_indices and sensory_indices must be non-empty")
        if system & sensory:
            raise ValueError(
                "system_indices and sensory_indices must be disjoint; "
                f"got overlap {sorted(system & sensory)}"
            )
        _validate_binary_substrate(self.substrate)
        _validate_sorted_indices("system_indices", self.system_indices)
        _validate_sorted_indices("sensory_indices", self.sensory_indices)

    @property
    def environment_indices(self) -> tuple[int, ...]:
        system = set(self.system_indices)
        return tuple(i for i in self.substrate.node_indices if i not in system)

    @property
    def node_labels(self):
        return self.substrate.node_labels

    @staticmethod
    def _validate_tau(tau, tau_clamp):
        if not isinstance(tau, int) or not isinstance(tau_clamp, int):
            raise ValueError("tau and tau_clamp must be integers")
        if tau < 1:
            raise ValueError(f"tau must be >= 1; got {tau}")
        if not 0 <= tau_clamp <= tau:
            raise ValueError(f"require 0 <= tau_clamp <= tau; got {tau_clamp}, {tau}")

    def triggered_tpm(self, *, tau, tau_clamp) -> TriggeredTPM:
        """The fixed-lag response distribution Pr(Sₜ | ∂S_{t−τ} = x).

        Parameters
        ----------
        tau : int
            The lag τ >= 1 at which the stimulus's effect is evaluated.
        tau_clamp : int
            The number of initial steps for which the sensory interface is held
            at the stimulus before being marginalized. Must satisfy
            ``0 <= tau_clamp <= tau``.

        Returns
        -------
        TriggeredTPM
            One system-state distribution per stimulus.
        """
        self._validate_tau(tau, tau_clamp)
        return build_triggered_tpm(
            self.substrate,
            self.sensory_indices,
            self.system_indices,
            tau=tau,
            tau_clamp=tau_clamp,
        )

    def triggered_states(self, *, tau, tau_clamp) -> dict:
        """Mapping ``{stimulus: response_state}`` over all stimuli.

        The response state is the most-probable (argmax) system state for each
        stimulus, the triggered state the Φ-structure computation consumes.
        """
        ttpm = self.triggered_tpm(tau=tau, tau_clamp=tau_clamp)
        return {
            x: ttpm.argmax_state(x) for x in utils.all_states(len(self.sensory_indices))
        }

    def triggered_state(self, stimulus, *, tau, tau_clamp) -> tuple[int, ...]:
        """The response state (argmax system state) for a single stimulus."""
        ttpm = self.triggered_tpm(tau=tau, tau_clamp=tau_clamp)
        return ttpm.argmax_state(tuple(stimulus))
