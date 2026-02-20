"""Substrate class for building PyPhi networks from units.

A Substrate is a stateless collection of Units. State is passed explicitly
when computing TPMs, Networks, or Subsystems.
"""

from functools import cached_property

import numpy as np
from tqdm.auto import tqdm

from .. import utils
from ..labels import NodeLabels
from ..network import Network
from ..subsystem import Subsystem
from ..tpm import ExplicitTPM
from .unit import CompositeUnit
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

    def compute_tpm(
        self,
        state: tuple[int, ...],
        input_state: tuple[int, ...] | None = None,
    ) -> ExplicitTPM:
        """Compute the substrate TPM for the given state and input state.

        The mechanisms are configured according to ``state`` and
        ``input_state``, then transition probabilities are computed for every
        possible ``from_state``.

        Args:
            state: The current state of the substrate. Determines each unit's
                own binary state for state-dependent mechanisms.
            input_state: The input state of the substrate. Determines each
                unit's input states for input-dependent mechanisms. Defaults
                to ``state``.

        Returns:
            The substrate's TPM as an ExplicitTPM.
        """
        if input_state is None:
            input_state = state
        all_from_states = list(utils.all_states(len(self._units)))
        show_progress = len(all_from_states) > PROGRESS_BAR_THRESHOLD
        rows = [
            self._combine_unit_tpms(from_state, state, input_state)
            for from_state in (
                tqdm(all_from_states) if show_progress else all_from_states
            )
        ]
        return ExplicitTPM(reshape_to_md(np.array(rows)))

    def _combine_unit_tpms(
        self,
        from_state: tuple[int, ...],
        state: tuple[int, ...],
        input_state: tuple[int, ...],
    ) -> list[float]:
        """Combine unit TPMs for a single transition row.

        Args:
            from_state: The state being transitioned from (indexes into unit
                TPMs to select the row).
            state: The current state of the substrate (determines each unit's
                own binary state for state-dependent mechanisms).
            input_state: The input state of the substrate (determines each
                unit's input states for input-dependent mechanisms).

        Returns:
            Activation probabilities for each unit.
        """
        probs = []
        for unit in self._units:
            unit_state = state[unit.index]
            unit_input_state = tuple(input_state[i] for i in unit.inputs)
            unit_tpm = unit.compute_tpm(unit_state, unit_input_state)
            pp = unit_tpm[tuple(from_state[i] for i in unit.inputs)]
            if not isinstance(pp, (int, float, np.number)):
                probs.append(float(pp[0]))
            else:
                probs.append(float(pp))
        return probs

    @cached_property
    def dynamic_tpm(self) -> ExplicitTPM:
        """The dynamic TPM of the substrate.

        For each row, the substrate is configured as if it is in that row's
        state: ``from_state = state = input_state = s``.
        """
        all_states = list(utils.all_states(len(self._units)))
        show_progress = len(all_states) > PROGRESS_BAR_THRESHOLD
        rows = [
            self._combine_unit_tpms(s, s, s)
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

    def network(
        self,
        state: tuple[int, ...],
        input_state: tuple[int, ...] | None = None,
    ) -> Network:
        """Get a Network for the given state.

        Args:
            state: The current state of the substrate.
            input_state: The input state of the substrate. Defaults to
                ``state``.

        Returns:
            A Network with the substrate's TPM conditioned on the given state.
        """
        return Network(self.compute_tpm(state, input_state), self.cm, self.node_labels)

    def subsystem(
        self,
        state: tuple[int, ...],
        input_state: tuple[int, ...] | None = None,
        nodes: tuple[int, ...] | None = None,
    ) -> Subsystem:
        """Get a Subsystem for the given state.

        Args:
            state: The current state of the substrate.
            input_state: The input state of the substrate. Defaults to
                ``state``.
            nodes: The node indices to include. Defaults to all nodes.

        Returns:
            A Subsystem conditioned on the given state.
        """
        if nodes is None:
            nodes = self.node_indices
        return Subsystem(self.network(state, input_state), state, nodes)

    def to_json(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "units": [u.to_json() for u in self._units],
        }


def create_substrate(
    node_params: dict,
    labels: list[str] | None = None,
    implicit: bool = False,
) -> Substrate:
    """Create a Substrate from a dictionary of per-node parameters.

    This is a convenience factory that handles inferring inputs from
    connectivity matrices and extracting input weights from weight matrices.

    Args:
        node_params: Per-node params, indexed by node index or label.
            Each entry should specify at minimum:
                - ``"mechanism"``: str
                - ``"inputs"``: tuple of input indices, or a connectivity
                  matrix (np.ndarray) from which inputs are inferred
                - ``"params"``: dict of mechanism parameters. If
                  ``"input_weights"`` is an np.ndarray (weight matrix),
                  weights are extracted automatically.
                - ``"composite"``: list of sub-unit dicts (optional).
                  Each sub-unit dict has the same structure as above.
                - ``"mechanism_combination"``: str (optional, for composite
                  units; defaults to ``"selective"``).
        labels: Optional list of labels for nodes. Defaults to string indices.
        implicit: Whether to use implicit (factored) TPM representation.

    Returns:
        A Substrate built from the given parameters.
    """
    n = len(node_params)
    labels = labels or [str(i) for i in range(n)]

    units = []
    for i in range(n):
        label = labels[i]
        p = node_params.get(i) or node_params.get(label)
        if p is None:
            raise ValueError(f"No parameters for node {i}/{label}")

        if "composite" in p:
            sub_units = []
            for sub in p["composite"]:
                sub_inputs = _resolve_inputs(sub.get("inputs"), i)
                params = _resolve_params(dict(sub["params"]), sub_inputs, i)
                sub_units.append(
                    Unit(
                        index=i,
                        label=label,
                        inputs=tuple(sub_inputs),
                        mechanism=sub["mechanism"],
                        params=params,
                    )
                )
            u = CompositeUnit(
                index=i,
                label=label,
                units=tuple(sub_units),
                mechanism_combination=p.get("mechanism_combination", "selective"),
            )
        else:
            inputs = _resolve_inputs(p.get("inputs"), i)
            params = _resolve_params(dict(p["params"]), inputs, i)
            u = Unit(
                index=i,
                label=label,
                inputs=tuple(inputs),
                mechanism=p["mechanism"],
                params=params,
            )

        units.append(u)

    return Substrate(tuple(units), implicit=implicit)


def _resolve_inputs(
    inputs_or_cm: tuple | list | np.ndarray, node_index: int
) -> list[int]:
    """Extract input indices from either explicit inputs or a connectivity matrix."""
    if isinstance(inputs_or_cm, np.ndarray):
        return list(np.nonzero(inputs_or_cm[:, node_index])[0])
    return list(inputs_or_cm)


def _resolve_params(params: dict, inputs: list[int], node_index: int) -> dict:
    """Extract per-node input weights from a weight matrix if needed."""
    if "input_weights" in params and isinstance(params["input_weights"], np.ndarray):
        W = params["input_weights"]
        params["input_weights"] = tuple(W[j, node_index] for j in inputs)
    return params
