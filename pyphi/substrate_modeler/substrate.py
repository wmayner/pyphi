"""Substrate class for building PyPhi networks from units.

A Substrate is a stateless collection of Units. State is passed explicitly
when computing TPMs, Networks, or Subsystems.
"""

from functools import cached_property

from tqdm.auto import tqdm
import numpy as np

from .. import utils
from ..tpm import ExplicitTPM
from ..labels import NodeLabels
from ..network import Network
from ..subsystem import Subsystem
from .unit import Unit
from .utils import reshape_to_md


PROGRESS_BAR_THRESHOLD = 2**20


class Substrate:
    """A stateless collection of Units defining a substrate's dynamics.

    Units define each node's mechanism, inputs, and parameters. State is
    passed explicitly to methods that need it (compute_tpm, network, subsystem).

    Args:
        units: The units that make up this substrate.
        implicit: Whether to use implicit (factored) TPM representation.
    """

    def __init__(
        self,
        units: tuple[Unit, ...],
        implicit: bool = False,
    ):
        self._units = units
        self._implicit = implicit

    @property
    def units(self) -> tuple[Unit, ...]:
        """The units that make up this substrate."""
        return self._units

    @cached_property
    def node_indices(self) -> tuple[int, ...]:
        """The indices of the units in the substrate."""
        return tuple(unit.index for unit in self._units)

    @cached_property
    def node_labels(self) -> NodeLabels:
        """The labels of the units in the substrate."""
        return NodeLabels([unit.label for unit in self._units], self.node_indices)

    def compute_tpm(self, state: tuple[int, ...]) -> ExplicitTPM:
        """Compute the substrate TPM for the given present state.

        Args:
            state: The present state of all units in the substrate.

        Returns:
            The substrate's TPM as an ExplicitTPM.
        """
        all_past_states = list(utils.all_states(len(self._units)))
        show_progress = len(all_past_states) > PROGRESS_BAR_THRESHOLD
        rows = [
            self._combine_unit_tpms(past_state, state)
            for past_state in (
                tqdm(all_past_states) if show_progress else all_past_states
            )
        ]
        return ExplicitTPM(reshape_to_md(np.array(rows)))

    def _combine_unit_tpms(
        self,
        past_state: tuple[int, ...],
        present_state: tuple[int, ...],
    ) -> list[float]:
        """Combine unit TPMs for a single past/present state pair.

        Args:
            past_state: The state at the previous time step.
            present_state: The state at the current time step.

        Returns:
            Activation probabilities for each unit.
        """
        probs = []
        for unit in self._units:
            unit_tpm = unit.state_dependent_tpm(present_state)
            pp = unit_tpm[tuple(past_state[i] for i in unit.inputs)]
            if not isinstance(pp, (int, float, np.number)):
                probs.append(float(pp[0]))
            else:
                probs.append(float(pp))
        return probs

    @cached_property
    def dynamic_tpm(self) -> ExplicitTPM:
        """The state-independent (dynamic) TPM of the substrate.

        For each state, the present state equals the past state.
        """
        all_states = list(utils.all_states(len(self._units)))
        show_progress = len(all_states) > PROGRESS_BAR_THRESHOLD
        rows = [
            self._combine_unit_tpms(s, s)
            for s in (tqdm(all_states) if show_progress else all_states)
        ]
        return ExplicitTPM(reshape_to_md(np.array(rows)))

    @cached_property
    def cm(self) -> np.ndarray:
        connectivity = np.zeros((len(self.node_indices), len(self.node_indices)))

        for unit in self._units:
            connectivity[unit.inputs, unit.index] = 1

        return connectivity

    def __repr__(self) -> str:
        return "Substrate({})".format("|".join(self.node_labels))

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self._units)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Substrate):
            return NotImplemented
        return self._units == other._units

    def network(self, state: tuple[int, ...]) -> Network:
        """Get a Network for the given state.

        Args:
            state: The present state of the substrate.

        Returns:
            A Network with the substrate's TPM conditioned on the given state.
        """
        return Network(self.compute_tpm(state), self.cm, self.node_labels)

    def subsystem(
        self,
        state: tuple[int, ...],
        nodes: tuple[int, ...] | None = None,
    ) -> Subsystem:
        """Get a Subsystem for the given state.

        Args:
            state: The present state of the substrate.
            nodes: The node indices to include. Defaults to all nodes.

        Returns:
            A Subsystem conditioned on the given state.
        """
        if nodes is None:
            nodes = self.node_indices
        return Subsystem(self.network(state), state, nodes)

    def to_json(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "units": [u.to_json() for u in self._units],
        }
