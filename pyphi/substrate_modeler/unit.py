"""
unit.py
=============
This module provides functionality for creating units that constitute substrate models in Integrated information theory.
The 'unit' module is separated into three sections:


Section 1 - Unit classes
------------------------
Defines three classes (and a helper class)
        HelperClass -- TPMdict
    Class 1 -- BaseUnit
    Class 2 -- Unit
    Class 3 -- CompositeUnit
Please refer to the docstrings for information about each of these.

Section 2 - Unit (I/O) functions
--------------------------------
This section contains functions for creating unit TPMs. That is, it provides specific, predifined functions that can be used to define units.

Section 3 - Unit validation
---------------------------
This section contains functions for validating the creation of units and their TPMs.

"""

# TODO:
# - Understand what is wrong with composite units. Something with TPM and indexing of inputs
# - allow for non-binary units
# - Deal with modulation (put into unit params?)
# - Create validation for units!


from typing import Union, Tuple, List, Callable

import numpy as np
import pyphi

from pyphi.tpm import ExplicitTPM

from .unit_functions import UNIT_FUNCTIONS
from .mechanism_combinations import MECHANISM_COMBINATIONS


class TPMDict(dict):
    """
    A dictionary-like class for caching transition probability matrices (TPMs)
    associated with a given `unit`.

    Parameters:
    -----------
    unit : object
        An object with a `tpm` method that returns a TPM associated with the current
        `unit` state and `input` state.

    Methods:
    --------
    __getitem__(self, substrate_state):
        Return the TPM associated with the given `substrate_state`.
        If the TPM is not already cached, calculate it using the inferred `unit` state
        and `input` state.

    __missing__(self, state):
        Calculate the TPM associated with the given `state` (a tuple containing the `unit`
        state and `input` state), store it for future use, and return it.

    Attributes:
    -----------
    unit : object
        An object with a `tpm` method that returns a TPM associated with the current
        `unit` state and `input` state.
    """

    def __init__(self, unit):
        """
        Initialize the TPMDict object.

        Parameters:
        -----------
        unit : object
            An object with a `tpm` method that returns a TPM associated with the
            `unit` state and `input` state inferred from the substrate state key.
        """
        self.unit = unit

    def __getitem__(self, substrate_state):
        """
        Return the TPM associated with the given `substrate_state`.
        If the TPM is not already cached, calculate it using the inferred `unit` state
        and `input` state.

        Parameters:
        -----------
        substrate_state : tuple
            A tuple containing the substrate state.

        Returns:
        --------
        tpm : numpy.ndarray
            The transition probability matrix associated with the given `substrate_state`.
        """
        # infer the unit state and input state from substrate state
        unit_state = (substrate_state[self.unit.index],)
        input_state = tuple([substrate_state[i] for i in self.unit.inputs])

        try:
            return super().__getitem__((unit_state, input_state))
        except KeyError:
            return self.__missing__((unit_state, input_state))

    def __missing__(self, state):
        """
        Calculate the TPM associated with the given 'state' (a tuple containing the `unit`
        state and `input` state), store it for future use, and return it.

        Parameters:
        -----------
        state : tuple
            A tuple containing the `unit` state and `input` state.

        Returns:
        --------
        tpm : numpy.ndarray
            The transition probability matrix associated with the given `state`.
        """
        # store original states, so as to reset it after extracting the tpm
        orig_state = self.unit.state, self.unit.input_state

        # set unit state
        self.unit.state = state[0]

        # set input state
        self.unit.input_state = state[1]

        # create tpm
        tpm = self.unit.tpm

        # cache the value for future use
        self[state] = tpm

        # reset unit state
        self.unit.state = orig_state[0]

        # reset input state
        self.unit.input_state = orig_state[1]

        return tpm


class BaseUnit:
    """
    Represents a basic unit in a substrate, with a binary state and input connections.

    Args:
        index (int): The index of the unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the unit, either as an int (0 or 1) or a tuple of ints.
        label (str, optional): A label for the unit. Defaults to None.
        inputs (tuple[int], optional): A tuple of indices of units that input to this unit. Defaults to (None,).
        input_state (tuple[int], optional): The binary state of the input units, as a tuple of ints. Defaults to (None,).

    Attributes:
        index (int): The index of the unit in the substrate.
        state (Tuple[int,]): The binary state of the unit, as a tuple (length 1) or int.
        label (str): A string label for the unit.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
    """

    def __init__(
        self,
        index: int,
        state: Union[int, Tuple[int,]],
        label: str = None,
        inputs: tuple[int] = (None,),
        input_state: tuple[int] = (None,),
        state_space: tuple[str] = (0, 1),
        original_index: int = False,
    ):

        # This unit's index in the list of units.
        self._index = index

        # Node labels used in the system
        if label is None:
            label = str(index)
        self._label = label

        # List of indices that input to the unit (one pr mechanism).
        self._inputs = inputs

        # Set unit state
        self._state = state
        if type(state) == int:
            if self._state not in (0, 1):
                raise ValueError("state must be 0 or 1")
            self._state = (state,)
        else:
            if state[0] not in (0, 1):
                raise ValueError("state must be 0 or 1")

        #  and input state
        if not all([s in (0, 1) for s in input_state]):
            raise ValueError("all input states must be 0 or 1")
        self._input_state = input_state

        self.state_space = state_space

        if not original_index:
            original_index = index
        self._original_index = original_index

    @property
    def index(self):
        """int: The index of the unit in the list of units."""
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        self.index

    @property
    def state(self):
        """
        Tuple[int,]: The binary state of the unit, as a tuple of ints.

        Raises:
            ValueError: If the state is not a valid binary value (0 or 1).
        """
        if type(self._state) is int:
            return (self._state,)
        else:
            return self._state

    @state.setter
    def state(self, state: Union[int, Tuple[int,]]):
        self._state = state
        self.state

    @property
    def label(self):
        """str: A label for the unit."""
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label
        self.label

    @property
    def inputs(self):
        """tuple[int]: A tuple of indices of units that input to this unit."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: tuple[int]):
        """tuple[int]: A tuple of indices of units that input to this unit."""
        self._inputs = inputs
        self.inouts

    @property
    def input_state(self):
        """
        tuple[int]: The binary state of the input units, as a tuple of ints.
        """
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: tuple[int]):
        """
        Set the binary state of the input units.

        Args:
            input_state (tuple[int]): The binary state of the input units, as a tuple of ints.

        Raises:
            ValueError: If any of the input states are not a valid binary value (0 or 1).
        """
        self._input_state = input_state
        self.input_state

    @property
    def original_index(self):
        """int: The index of the unit in the list of units."""
        return self._original_index

    @original_index.setter
    def original_index(self, index: int):
        self._original_index = index
        self.original_index

    def __repr__(self):
        """
        Return a string representation of the unit.
        """
        return "Unit(label={}, state={})".format(self.label, self.state)

    def __str__(self):
        """
        Return a string representation of the unit.
        """
        return self.__repr__()


class Unit(BaseUnit):
    """
    Represents a functional unit in a system, with a binary state and input connections,
    as well as additional parameters specific to the unit's input-output mechanism.

    Args:
        index (int): The index of the unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the unit, either as an int (0 or 1) or a tuple of ints.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
        mechanism (str): The type of unit mechanism (e.g., 'and', 'or', etc. see unit_functions).
        params (dict): A dictionary of parameters specifying to the unit's mechanism.
        label (str, optional): A label for the unit. Defaults to None.

    Attributes:
        index (int): The index of the unit in the list of units.
        state (Tuple[int,]): The binary state of the unit, as a tuple of ints.
        label (str): A label for the unit.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
        params (dict): A dictionary of parameters specific to the unit's mechanism.
        mechanism (str): The type of unit mechanism (e.g., 'and', 'or', etc.).
        tpm (numpy.ndarray): The truth table of the unit, as a numpy array of binary values.
    """

    def __init__(
        self,
        index: int,
        inputs: tuple[int],
        mechanism: Union[Callable, str, np.array],
        state: Union[int, Tuple[int,]] = None,
        state_space: Tuple[str] = (0, 1),
        input_state: tuple[int] = None,
        params: dict = None,
        label: str = None,
        unit_type: str = "custom",
        original_index: int = False,
    ):
        if type(state) is int:
            state = (state,)
        super().__init__(
            index=index,
            state=state,
            state_space=state_space,
            label=label,
            inputs=inputs,
            input_state=input_state,
            original_index=original_index,
        )

        # Store the parameters
        self._params = params

        # Store the type of unit and get mechanism function
        if isinstance(mechanism, str):
            unit_type = str(mechanism)
            mechanism = UNIT_FUNCTIONS[mechanism]

        self._type = unit_type
        self._mechanism = mechanism

        # set tpm
        self.tpm

        # initialize the local storage variable for unit TPM
        self._state_dependent_tpm = TPMDict(self)

        # validate unit
        assert self.validate(), "Unit did not pass validation"

    @property
    def params(self):
        """
        dict: A dictionary of parameters specific to the unit's mechanism.
        """
        return self._params

    @property
    def mechanism(self):
        """
        str: The type of unit mechanism (e.g., 'and', 'or', etc.).
        """
        return self._mechanism

    @property
    def tpm(self):
        """
        numpy.ndarray: The truth table of the unit, as a numpy array of binary values.
        """
        if self._params is None:
            if type(self.mechanism) is np.ndarray:
                self._tpm = self.mechanism
            else:
                self._tpm = None
        else:
            self._tpm = self.mechanism(self, **self.params)
        return ExplicitTPM(self._tpm)

    def state_dependent_tpm(self, substrate_state: tuple[int]):
        """
        Set the binary state of the input units and return the truth table of the unit.

        Args:
            substrate_state (tuple[int]): The binary state of the substrate units, as a tuple of ints.

        Returns:
            numpy.ndarray: The truth table of the unit, as a numpy array of binary values.
        """
        return self._state_dependent_tpm[substrate_state]

    def validate(self):
        """Return whether the specifications for the unit are valid.

        The checks for validity are defined in the local UNIT_FUNCTIONS object.
        """
        return True

    def __repr__(self):
        return "Unit(type={}, label={}, state={})".format(
            self._type, self.label, self.state
        )

    def __eq__(self, other):
        """Return whether this unitis identical to another.

        Two nodes are equal if they have the same TPMs, states, and inputs.

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
        )

    def __copy__(self):
        return Unit(
            self.index,
            self.inputs,
            input_state=self.input_state,
            mechanism=self.mechanism,
            params=self.params,
            label=self.label,
            state=self.state,
        )

    # TODO?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class CompositeUnit(Unit):
    """
    Represents a composite unit in a system, composed of multiple individual units.

    Args:
        index (int): The index of the composite unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the composite unit, either as an int (0 or 1) or a tuple of ints.
        units (List[Unit]): A list of individual units that compose the composite unit.
        label (str, optional): The label for the composite unit.
        mechanism_combination (Union[str, np.ndarray, dict], optional): The mechanism(s) used to combine the individual units' truth tables into a composite truth table.

    Attributes:
        inputs (tuple[int]): The indices of the input units to the composite unit.
        input_state (tuple[int]): The binary states of the input units to the composite unit.
        tpm (numpy.ndarray): The truth table of the composite unit, as a numpy array of binary values.

    Methods:

    """

    def __init__(
        self,
        index: int,
        units: List[Unit],
        state: Union[int, Tuple[int,]],
        label: str = None,
        mechanism_combination: Union[str, np.ndarray, dict] = None,
    ):
        # store the list of `Unit` objects that make up this `CompositeUnit`
        self.units = units

        # Store the waythe tpms from the component `Unit`s combine to give the composit I/O function
        if isinstance(mechanism_combination, str):
            mechanism_combination = MECHANISM_COMBINATIONS[mechanism_combination]
        self._mechanism_combination = mechanism_combination

        # Determine the input indices of the `CompositeUnit`
        self.inputs

        # Determine the input state of the `CompositeUnit`
        self.input_state

        if type(state) is int:
            state = (state,)
        # Initialize the unit object
        super().__init__(
            index=index,
            state=state,
            inputs=self._inputs,
            input_state=self._input_state,
            mechanism="composite",  #: {}'.format('+'.join([unit.mechanism for unit in self.units])),
            params=None,
            label=label,
        )

        # get the TPM of the composite unit
        self.tpm

    @property
    def inputs(self):
        """
        tuple[int]: The indices of the input units to the composite unit.
        """
        all_inputs = tuple(set([ix for unit in self.units for ix in unit._inputs]))
        self._inputs = tuple(sorted(all_inputs))
        return self._inputs

    @property
    def state(self):
        """
        tuple[int]: The binary states of the input units to the composite unit.
        """
        self._state = (
            np.max(
                [
                    unit.state[0] if type(unit.state) is tuple else unit.state
                    for unit in self.units
                ]
            ),
        )
        return self._state

    @state.setter
    def state(self, unit_state: Union[Tuple[int], int]):
        """
        Set the binary states of the input units to the composite unit.

        Args:
            input_state (tuple[int]): The binary states of the input units, as a tuple of ints.
        """
        if type(unit_state) is int:
            unit_state = (unit_state,)
        # update state of units
        for unit in self.units:
            unit.state = unit_state
        self._state = unit_state
        self.state

    @property
    def input_state(self):
        """
        tuple[int]: The binary states of the input units to the composite unit.
        """
        state_dict = dict()
        for unit in self.units:
            for ix, state in zip(unit._inputs, unit._input_state):
                # NOTE! if there are conflicts, the latter will be used
                state_dict[ix] = state

        self._input_state = tuple([state_dict[ix] for ix in self._inputs])
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: Tuple[int]):
        """
        Set the binary states of the input units to the composite unit.

        Args:
            input_state (tuple[int]): The binary states of the input units, as a tuple of ints.
        """
        # update state of units
        for unit in self.units:
            unit.input_state = tuple(
                [
                    state
                    for state, i in zip(input_state, self._inputs)
                    if i in unit._inputs
                ]
            )
        self._input_state = input_state
        self.input_state

    @property
    def tpm(self):
        """
        numpy.ndarray: The truth table of the composite unit, as a numpy array of binary values.
        """
        tpms = [unit.tpm for unit in self.units]
        return self.combine_unit_tpms(tpms)

    def combine_unit_tpms(self, tpms):
        # Check this
        expanded_tpm = self.expand_tpms(tpms)

        # combine subunit TPMs into composite unit tpm
        return self.apply_tpm_combination(expanded_tpm)

    def expand_tpms(self, tpms):
        def get_subset_state(state, subset_indices):
            """tuple[int (binary)]: the state of a subset of indices.

            Args:
                state (tuple[int(binary)]): The (binary) state of the full set of inputs.
                subset_indices (tuple[int]): The indices (relative to the state) for the subset.
            """
            return tuple([state[ix] for ix in subset_indices])

        expanded_tpms = []
        for tpm, unit in zip(tpms, self.units):

            # get indices of unit inputs among the composite unit inputs
            unit_specific_inputs = tuple(
                [self._inputs.index(i) for i in self._inputs if i in unit._inputs]
            )

            # get mechanism activation probabilities for all potential input states
            mechanism_activation = []
            for state in pyphi.utils.all_states(len(self._inputs)):
                P = tpm[get_subset_state(state, unit_specific_inputs)]

                if not type(P) in (float, np.float64):  # np.ndarray:
                    P = P[0]
                mechanism_activation.append(P)

            expanded_tpms.append(mechanism_activation)

        # make the TPMs into an array of correct shape
        return np.array(expanded_tpms).T

    def apply_tpm_combination(self, expanded_tpms):
        return ExplicitTPM(self._mechanism_combination(expanded_tpms))

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index
