"""Unit classes for substrate modeling.

This module provides the Unit and CompositeUnit classes for defining nodes
in substrate models. Units define a node's dynamics (mechanism, inputs, params)
without holding state - state is passed explicitly when computing TPMs.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..tpm import ExplicitTPM
from .. import utils

from .unit_functions import UNIT_FUNCTIONS
from .mechanism_combinations import MECHANISM_COMBINATIONS


class TPMCache(dict):
    """Cache for computed unit TPMs, keyed by (unit_state, input_state).

    This is a pure cache with no side effects - it simply memoizes calls to
    unit.compute_tpm().
    """

    def __init__(self, unit: "Unit"):
        super().__init__()
        self._unit = unit

    def __getitem__(self, substrate_state: tuple[int, ...]) -> ExplicitTPM:
        """Get the TPM for the given substrate state, computing if needed.

        Args:
            substrate_state: The full state of all units in the substrate.

        Returns:
            The unit's TPM for its state and input state extracted from
            the substrate state.
        """
        unit_state = substrate_state[self._unit.index]
        input_state = tuple(substrate_state[i] for i in self._unit.inputs)
        key = (unit_state, input_state)

        if key not in self:
            self[key] = self._unit.compute_tpm(unit_state, input_state)
        return super().__getitem__(key)


class Unit:
    """A unit in a substrate model.

    Represents a node's dynamics: its mechanism (I/O function), inputs, and
    parameters. Units are stateless - state is passed explicitly when computing
    TPMs.

    Args:
        index: The unit's index in the substrate.
        inputs: Indices of units that provide input to this unit.
        mechanism: Either a string naming a built-in mechanism (e.g., 'and',
            'sigmoid'), a callable mechanism function, or a numpy array
            representing a raw TPM.
        params: Parameters passed to the mechanism function.
        label: Display label for the unit. Defaults to str(index).
        original_index: The unit's original index before any re-indexing.
            Defaults to index.
    """

    def __init__(
        self,
        index: int,
        inputs: tuple[int, ...],
        mechanism: Callable | str | NDArray[np.floating],
        params: dict | None = None,
        label: str | None = None,
        original_index: int | None = None,
    ):
        self._index = index
        self._inputs = tuple(inputs)
        self._label = label if label is not None else str(index)
        self._original_index = original_index if original_index is not None else index
        self._params = params if params is not None else {}

        # Resolve mechanism
        if isinstance(mechanism, str):
            self._mechanism_type = mechanism
            self._mechanism = UNIT_FUNCTIONS[mechanism]
        elif isinstance(mechanism, np.ndarray):
            self._mechanism_type = "raw_tpm"
            self._mechanism = mechanism
        else:
            self._mechanism_type = "custom"
            self._mechanism = mechanism

        # Initialize TPM cache
        self._tpm_cache = TPMCache(self)

    @property
    def index(self) -> int:
        """The unit's index in the substrate."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        self._index = value
        # Invalidate cache since index changed
        self._tpm_cache = TPMCache(self)

    @property
    def inputs(self) -> tuple[int, ...]:
        """Indices of units that provide input to this unit."""
        return self._inputs

    @property
    def label(self) -> str:
        """Display label for the unit."""
        return self._label

    @property
    def original_index(self) -> int:
        """The unit's original index before any re-indexing."""
        return self._original_index

    @property
    def mechanism(self) -> Callable | NDArray[np.floating]:
        """The mechanism function or raw TPM array."""
        return self._mechanism

    @property
    def mechanism_type(self) -> str:
        """String name of the mechanism type."""
        return self._mechanism_type

    @property
    def params(self) -> dict:
        """Parameters passed to the mechanism function."""
        return self._params

    def compute_tpm(self, state: int, input_state: tuple[int, ...]) -> ExplicitTPM:
        """Compute this unit's TPM for the given state.

        Args:
            state: This unit's current binary state (0 or 1).
            input_state: Current states of this unit's input units.

        Returns:
            The unit's transition probability matrix.
        """
        if isinstance(self._mechanism, np.ndarray):
            return ExplicitTPM(self._mechanism)
        return ExplicitTPM(
            self._mechanism(self, state, input_state, **self._params)
        )

    def state_dependent_tpm(self, substrate_state: tuple[int, ...]) -> ExplicitTPM:
        """Get the TPM for the given substrate state (cached).

        Args:
            substrate_state: The full state of all units in the substrate.

        Returns:
            The unit's TPM for its state and input state extracted from
            the substrate state.
        """
        return self._tpm_cache[substrate_state]

    def __repr__(self) -> str:
        return f"Unit(type={self._mechanism_type}, label={self._label}, inputs={self._inputs})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return (
            self._mechanism_type == other._mechanism_type
            and self._inputs == other._inputs
            and self._params == other._params
            # For raw TPM mechanisms, compare the arrays
            and (
                np.array_equal(self._mechanism, other._mechanism)
                if isinstance(self._mechanism, np.ndarray)
                else self._mechanism == other._mechanism
            )
        )

    def __hash__(self) -> int:
        # Hash based on immutable structural properties
        return hash((self._mechanism_type, self._inputs, tuple(self._params.items())))

    def __copy__(self) -> "Unit":
        if isinstance(self._mechanism, np.ndarray):
            mechanism = np.array(self._mechanism)
        elif self._mechanism_type != "custom":
            mechanism = self._mechanism_type
        else:
            mechanism = self._mechanism
        return Unit(
            index=self._index,
            inputs=self._inputs,
            mechanism=mechanism,
            params=self._params.copy() if self._params else None,
            label=self._label,
            original_index=self._original_index,
        )

    def to_json(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "index": self._index,
            "inputs": self._inputs,
            "mechanism_type": self._mechanism_type,
            "params": self._params,
            "label": self._label,
            "original_index": self._original_index,
        }


class CompositeUnit(Unit):
    """A unit composed of multiple sub-units.

    The composite unit's TPM is computed by combining the TPMs of its
    sub-units according to a specified combination strategy.

    Args:
        index: The composite unit's index in the substrate.
        units: The sub-units that make up this composite unit.
        label: Display label for the unit.
        mechanism_combination: How to combine sub-unit TPMs. Either a string
            naming a built-in combination (e.g., 'selective', 'average'),
            or a callable.
        original_index: The unit's original index before any re-indexing.
    """

    def __init__(
        self,
        index: int,
        units: tuple[Unit, ...],
        label: str | None = None,
        mechanism_combination: str | Callable | None = None,
        original_index: int | None = None,
    ):
        self._sub_units = units

        # Resolve combination function
        if mechanism_combination is None:
            self._mechanism_combination = MECHANISM_COMBINATIONS["selective"]
        elif isinstance(mechanism_combination, str):
            self._mechanism_combination = MECHANISM_COMBINATIONS[mechanism_combination]
        else:
            self._mechanism_combination = mechanism_combination

        # Inputs are the union of all sub-unit inputs
        all_inputs = sorted(set(i for unit in units for i in unit.inputs))

        super().__init__(
            index=index,
            inputs=tuple(all_inputs),
            mechanism="composite",  # Marker; compute_tpm is overridden
            label=label,
            original_index=original_index,
        )

    @property
    def sub_units(self) -> tuple[Unit, ...]:
        """The sub-units that make up this composite unit."""
        return self._sub_units

    def compute_tpm(self, state: int, input_state: tuple[int, ...]) -> ExplicitTPM:
        """Compute the composite TPM by combining sub-unit TPMs.

        Args:
            state: This unit's current binary state (0 or 1).
            input_state: Current states of this unit's input units.

        Returns:
            The composite unit's TPM.
        """
        tpms = []
        for sub_unit in self._sub_units:
            # Map composite input_state to sub-unit input_state
            sub_input_state = tuple(
                input_state[self._inputs.index(i)] for i in sub_unit.inputs
            )
            tpms.append(sub_unit.compute_tpm(state, sub_input_state))

        expanded = self._expand_tpms(tpms)
        return self._apply_tpm_combination(expanded)

    def _expand_tpms(self, tpms: list[ExplicitTPM]) -> NDArray[np.floating]:
        """Expand sub-unit TPMs to the full input space.

        Each sub-unit may have different inputs. This method expands each
        sub-unit's TPM so they all have the same shape, indexed by the
        composite unit's full input set.

        Args:
            tpms: List of TPMs from sub-units.

        Returns:
            Array of shape (n_states, n_sub_units) where n_states is 2^n_inputs.
        """
        def get_subset_state(
            full_state: tuple[int, ...], subset_indices: tuple[int, ...]
        ) -> tuple[int, ...]:
            """Extract the state of a subset of indices from a full state."""
            return tuple(full_state[ix] for ix in subset_indices)

        expanded_tpms = []
        for tpm, sub_unit in zip(tpms, self._sub_units):
            # Get indices of sub-unit inputs within composite unit inputs
            sub_unit_input_indices = tuple(
                self._inputs.index(i) for i in sub_unit.inputs
            )

            # Get activation probability for each possible input state
            mechanism_activation = []
            for full_state in utils.all_states(len(self._inputs)):
                sub_state = get_subset_state(full_state, sub_unit_input_indices)
                prob = tpm[sub_state]
                if not isinstance(prob, (int, float, np.number)):
                    prob = float(prob[0])
                mechanism_activation.append(float(prob))

            expanded_tpms.append(mechanism_activation)

        return np.array(expanded_tpms).T

    def _apply_tpm_combination(
        self, expanded_tpms: NDArray[np.floating]
    ) -> ExplicitTPM:
        """Apply the combination function to expanded TPMs.

        Args:
            expanded_tpms: Array of shape (n_states, n_sub_units).

        Returns:
            Combined TPM.
        """
        return ExplicitTPM(self._mechanism_combination(expanded_tpms))

    def __repr__(self) -> str:
        return (
            f"CompositeUnit(label={self._label}, "
            f"sub_units={[u.label for u in self._sub_units]}, "
            f"inputs={self._inputs})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeUnit):
            return NotImplemented
        return (
            self._sub_units == other._sub_units
            and self._mechanism_combination == other._mechanism_combination
        )

    def __hash__(self) -> int:
        return hash((self._sub_units, self._mechanism_combination))

    def to_json(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "index": self._index,
            "sub_units": [u.to_json() for u in self._sub_units],
            "label": self._label,
            "original_index": self._original_index,
        }
