# TODO
# - FIX INPUT STATE SETTING OF UNITS MAKE SURE THE STATE AND TPM OF UNITS ARE PROPERLY UPDATED

"""
substrate.py
=============
This module provides functionality for creating substrates from units for Integrated information theory analysis.
The 'substrate' module currently only has one section:


Section 1 - Classes
-------------------
    Class 1 -- Substrate
    Class 2 -- System
Please refer to the docstrings for information about each of these.

"""

from typing import Tuple, List, Union
from functools import cached_property

from .unit import Unit
from .utils import reshape_to_md
import pyphi
from pyphi.tpm import ExplicitTPM

from tqdm.auto import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


PROGRESS_BAR_THRESHOLD = 2**20


class Substrate:
    """A substrate of `Unit's.

    Attributes:
        units (List[Unit]): List of `Unit`s that make up the substrate.
        state (Tuple[int]): The binary state of the substrate units, as a tuple of ints.

    Properties:
        node_indices (Tuple[int]): The indices of the `Unit`s in the substrate, as a tuple of ints.
        node_labels (pyphi.labels.NodeLabels): The labels of the `Unit`s in the substrate.
        tpm (numpy.ndarray): The truth table of the substrate, as a numpy array of binary values.
        dynamic_tpm (numpy.ndarray): The dynamic truth table of the substrate, as a numpy array of binary values.
        cm (numpy.ndarray): The connectivity matrix of the substrate, as a numpy array of binary values.

    Methods:
        combine_unit_tpms(units, past_state, present_state): Combine the truth tables of `Unit`s in the substrate to compute the substrate's truth table.
        get_network(state=None): Get a `pyphi.network.Network` corresponding to the current or a specified state of the substrate.
        get_subsystem(state=None,nodes=None): Get a `pyphi.subsystem.Subsystem`

    """

    def __init__(
        self,
        units: List[Unit],
        state: Tuple[int] = None,
        input_state: Tuple[int] = None,
        implicit: bool = False,
    ):
        """Initialize a new `Substrate` object."""

        self._implicit = implicit
        self._units = units
        self.state = state
        if input_state is None:
            input_state = tuple(self._state)
        self.input_state = input_state

    @property
    def units(self):
        return self._units

    @property
    def state(self):
        """Tuple[int]: The binary state of the substrate units, as a tuple of ints."""

        if self._state is not None:
            for s, u in zip(self._state, self._units):
                u.state = s
        else:
            self._state = tuple(
                [u.state[0] if type(u.state) is tuple else u.state for u in self._units]
            )
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        self.state

    @property
    def input_state(self):
        """Tuple[int]: The binary input state to the substrate units, as a tuple of ints."""
        # set units to correct input state
        for u in self._units:
            u.input_state = tuple(self._input_state[i] for i in u._inputs)

        return self._input_state

    @input_state.setter
    def input_state(self, state):
        self._input_state = state
        self.input_state

    @cached_property
    def node_indices(self):
        """Tuple[int]: The indices of the `Unit`s in the substrate, as a tuple of ints."""

        return tuple([unit.index for unit in self._units])

    @cached_property
    def node_labels(self):
        """pyphi.labels.NodeLabels: The labels of the `Unit`s in the substrate."""

        return pyphi.labels.NodeLabels(
            [unit.label for unit in self._units], self.node_indices
        )

    @property
    def tpm(self):
        """numpy.ndarray: The truth table of the substrate, as a numpy array of binary values."""

        if self._implicit:
            # get the unit tpm for every unit in the substrate
            return [
                np.stack((1 - np.array(unit_tpm), np.array(unit_tpm)), axis=len(self))
                for unit_tpm in self.unit_tpms()
            ]
        else:
            # running through all possible input states
            all_states = list(pyphi.utils.all_states(len(self._units)))

            return reshape_to_md(
                np.array(
                    [
                        self.combine_unit_tpms(self._units, past_state, self.state)
                        for past_state in (
                            tqdm(all_states)
                            if len(all_states) > PROGRESS_BAR_THRESHOLD
                            else all_states
                        )
                    ]
                )
            )

    def state_dependent_tpm(self, state):
        """numpy.ndarray: The truth table of the substrate, as a numpy array of binary values."""
        self.state = state
        return self.tpm

    def combine_unit_tpms(self, units, past_state, present_state):
        """Combine the truth tables of `Unit`s in the substrate to compute the substrate's truth table.

        Args:
            units (List[Unit]): List of `Unit`s in the substrate.
            past_state (Tuple[int]): The binary state of the substrate at the previous time step, as a tuple of ints.
            present_state (Tuple[int]): The binary state of the substrate at the current time step, as a tuple of ints.

        Returns:
            List[float]: The combined truth table of the substrate, as a list of float values.
        """

        probs = []
        for unit in units:
            pp = unit.state_dependent_tpm(present_state)[
                tuple([past_state[i] for i in unit.inputs])
            ]
            if not isinstance(pp, (int, float, np.number)):
                probs.append(float(pp[0]))
            else:
                probs.append(pp)

        return probs
        """return [
            float(
                unit.state_dependent_tpm(present_state)[
                    tuple([past_state[i] for i in unit.inputs])
                ][0]
            )
            for unit in units
        ]"""

    @cached_property
    def dynamic_tpm(self):
        if self._implicit:
            print("Not implemented for implicit substrates")
            return False
        else:
            orig_state = self._state
            # running through all possible substrate states
            all_states = list(pyphi.utils.all_states(len(self._units)))

            dynamic_tpm = [
                self.combine_unit_tpms(self._units, state, state)
                for state in (
                    tqdm(all_states)
                    if len(all_states) > PROGRESS_BAR_THRESHOLD
                    else all_states
                )
            ]
            self.state = orig_state
            return ExplicitTPM(reshape_to_md(np.array(dynamic_tpm)))

    @cached_property
    def cm(self):
        connectivity = np.zeros((len(self.node_indices), len(self.node_indices)))

        for unit in self._units:
            connectivity[unit.inputs, unit.index] = 1

        return connectivity

    def __repr__(self):
        return "Substrate({})".format("|".join(self.node_labels))

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._units)

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def isolate_subset(
        self,
        subset_indices,
        indices_to_condition: Tuple[int] = False,
    ):
        units = [unit for i, unit in enumerate(self._units) if i in subset_indices]
        current_inputs = [unit.inputs for unit in units]
        indices_to_marginalize_out = [
            tuple(unit._inputs.index(i) for i in inputs if i not in subset_indices)
            for unit, inputs in zip(units, current_inputs)
        ]
        new_tpms = []
        if indices_to_condition:
            indices_to_condition = [
                {
                    unit._inputs.index(i): unit._input_state[unit._inputs.index(i)]
                    for i in inputs
                    if i in indices_to_condition and i not in subset_indices
                }
                for unit, inputs in zip(units, current_inputs)
            ]
            for unit, cond in zip(units, indices_to_condition):
                new_tpms.append(unit.tpm.condition_tpm(cond))
        new_index_mapping = {unit.index: i for i, unit in enumerate(units)}
        new_tpms = [
            np.squeeze(np.array(new_tpm.marginalize_out(marge)))[..., np.newaxis]
            for new_tpm, marge in zip(new_tpms, indices_to_marginalize_out)
        ]
        new_indices = [new_index_mapping[i] for i in subset_indices]
        old_indices = [i for i in subset_indices]
        new_inputs = [
            tuple(new_index_mapping[i] for i in inputs if i in subset_indices)
            for inputs in current_inputs
        ]
        new_state = tuple([self.state[i] for i in subset_indices])
        new_input_states = [
            tuple(self.input_state[i] for i in new_input) for new_input in new_inputs
        ]
        new_units = [
            Unit(
                index=unit_index,
                inputs=unit_inputs,
                mechanism=unit_tpm,
                state=unit_state,
                input_state=unit_input_state,
                label=unit.label,
                unit_type=unit._type,
                original_index=old_index,
            )
            for (
                unit_index,
                unit_inputs,
                unit_tpm,
                unit_state,
                unit_input_state,
                unit,
                old_index,
            ) in zip(
                new_indices,
                new_inputs,
                new_tpms,
                new_state,
                new_input_states,
                units,
                old_indices,
            )
        ]

        new_input_state = tuple([self.input_state[i] for i in subset_indices])

        return Substrate(
            units=new_units,
            state=new_state,
            input_state=new_input_state,
            implicit=self._implicit,
        )

    def network(self, state: Tuple[int] = None):
        # ensure state is set correctly
        self.state = state

        # return network based on implicit or explicit TPM
        if self._implicit:
            return pyphi.Network(
                self.tpm,
                cm=self.cm,
                node_labels=self.node_labels,
                state_space=self.unit_state_space(),
            )
        elif state is not None:
            return pyphi.network.Network(self.tpm, self.cm, self.node_labels)
        else:
            return pyphi.network.Network(self.dynamic_tpm, self.cm, self.node_labels)

    def subsystem(self, nodes: Tuple[int] = None):
        if nodes is None:
            nodes = self.node_indices
        return pyphi.Subsystem(
            self.network(self.state),
            self.state,
            nodes,
        )

    def expand_unit_tpm_dimensions(self, unit):
        # expand unit tpm to fit substrate size
        unit_tpm = unit.tpm
        input_indices = unit.inputs
        new_shape = np.ones(len(self), dtype=int)
        new_shape[list(input_indices)] = unit_tpm.shape[:-1]
        return unit_tpm.reshape(new_shape)

    def unit_tpms(self):
        # create unit tpms with dimensions expanded to substrate size
        return [self.expand_unit_tpm_dimensions(unit) for unit in self._units]

    def unit_state_space(self):

        # create unit tpms with dimensions expanded to substrate size
        return [tuple(unit.state_space) for unit in self._units]

    def concept(
        self,
        mechanism: Tuple[int],
        cause_purview_indices: Tuple[int] = False,
        effect_purview_indices: Tuple[int] = False,
        purview_max_size: int = 10,
        indices_to_condition: tuple[int] = False,
        cause_purviews: Tuple[Tuple[int]] = False,
        effect_purviews: Tuple[Tuple[int]] = False,
    ):
        if not cause_purview_indices:
            cause_purview_indices = tuple(
                set(
                    [
                        i
                        for ix in mechanism
                        for i, _input in enumerate(self.cm[:, ix])
                        if not _input == 0.0
                    ]
                )
            )
        if not effect_purview_indices:
            effect_purview_indices = tuple(
                set(
                    [
                        i
                        for ix in mechanism
                        for i, _output in enumerate(self.cm[ix, :])
                        if not _output == 0.0
                    ]
                )
            )

        # isolate the subset of units in the mechanism or purviews
        unit_indices = tuple(
            set([i for i in mechanism + cause_purview_indices + effect_purview_indices])
        )

        # define the smaller subsystem
        subset = self.isolate_subset(
            unit_indices, indices_to_condition=indices_to_condition
        )
        subsystem = subset.subsystem()

        index_mapping = {unit.original_index: i for i, unit in enumerate(subset.units)}

        # redefine mechanism
        mechanism = tuple([index_mapping[i] for i in mechanism])

        # define possible purviews
        cause_purviews = tuple(
            pyphi.utils.powerset(
                tuple([index_mapping[i] for i in cause_purview_indices]),
                max_size=purview_max_size,
                nonempty=True,
            )
        )
        effect_purviews = tuple(
            pyphi.utils.powerset(
                tuple([index_mapping[i] for i in effect_purview_indices]),
                max_size=purview_max_size,
                nonempty=True,
            )
        )
        print(len(cause_purviews), len(effect_purviews))
        # compute concept
        return subsystem.concept(
            mechanism, cause_purviews=cause_purviews, effect_purviews=effect_purviews
        )

    def model(self, state=None):
        rows, cols = np.where(self.cm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        gr.graph["params"] = "test"

        return gr

    def plot_model(self, state=None):

        gr = self.get_model(state=None)
        nodes = gr.nodes

        if state is None:
            state = (1,) * len(self.node_indices)

        nx.draw(
            gr,
            node_size=500,
            labels={i: l for i, l in enumerate(self.node_labels)},
            with_labels=True,
            node_color=["blue" if state[s] == 1 else "gray" for s in nodes],
        )
        plt.show()

    def simulate(self, initial_state=None, timesteps=1000, clamp=False, evoked=False):

        rng = np.random.default_rng(0)
        if not clamp:
            # Just simulating from initial state
            if initial_state == None:
                initial_state = tuple(rng.integers(0, 2, len(self)))
            states = [initial_state]
            for t in range(timesteps):
                P_next = self.dynamic_tpm[states[-1]]
                comparison = rng.random(len(initial_state))
                states.append(
                    tuple([1 if P > c else 0 for P, c in zip(P_next, comparison)])
                )

        else:
            # simulating with some units clamped to a state
            clamped_ix = list(clamp.keys())[0]
            clamped_state = list(clamp.values())[0]
            if not evoked:

                if initial_state is None:
                    initial_state = list(rng.integers(0, 2, len(self)))

                for ix, s in zip(clamped_ix, clamped_state):
                    initial_state[ix] = s
                states = [tuple(initial_state)]

                for t in range(timesteps):
                    P_next = self.dynamic_tpm[states[-1]]
                    comparison = rng.random(len(initial_state))

                    state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                    for ix, s in zip(clamped_ix, clamped_state):
                        state[ix] = s
                    states.append(tuple(state))
            elif type(evoked) == int:
                print("hey", flush=True)

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)

                    for i in clamped_ix:
                        initial_state.insert(i, np.random.randint(0, 2))

                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        if t >= evoked:
                            for ix, s in zip(clamped_ix, clamped_state):
                                state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)
            elif type(evoked) == list:

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)

                    for i in clamped_ix:
                        initial_state.insert(i, np.random.randint(0, 2))

                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        if t >= evoked[0] and t < evoked[1]:
                            for ix, s in zip(clamped_ix, clamped_state):
                                state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)

            elif evoked == "all":

                states = [(0,) * len(self)]

                for clamp in tqdm(list(pyphi.utils.all_states(len(clamped_ix)))):

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[states[-1]]
                        comparison = rng.random(len(self))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        for ix, s in zip(clamped_ix, clamp):
                            state[ix] = s

                        states.append(tuple(state))

            else:

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)
                    for i, s in zip(clamped_ix, clamped_state):
                        initial_state.insert(i, s)
                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        for ix, s in zip(clamped_ix, clamped_state):
                            state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)

        return states

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class System(pyphi.subsystem.Subsystem):
    def __init__(
        self,
        units: list[Unit],
        substrate_ixs: tuple[int],
        substrate_state: tuple[int],
        system_ixs: tuple[int],
    ):
        self.units = units
        self.state = tuple([unit.state[0] for unit in self.units])
        self.node_indices = tuple([unit.index for unit in self.units])
        self.node_labels = tuple([unit.label for unit in self.units])

        print(list(self.units[0].state_dependent_tpm.keys())[0], flush=True)
        system_units = self.subsystem_units(substrate_ixs, substrate_state, system_ixs)

        print(list(system_units[0].state_dependent_tpm.keys())[0], flush=True)
        system_state = tuple([unit.state[0] for unit in system_units])
        self.substrate = Substrate(system_units, state=system_state)
        self.subsystem = self.substrate.get_subsystem()

        self.units = system_units
        self.state = tuple([unit.state[0] for unit in self.units])
        self.node_indices = tuple([unit.index for unit in self.units])
        self.node_labels = tuple([unit.label for unit in self.units])

    def __repr__(self):
        return "System {} in {}".format(
            "|".join([u.label for u in self.units]),
            "".join([str(u.state[0]) for u in self.units]),
        )

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.units)

    def subsystem_units(
        self,
        substrate_ixs: tuple[int],
        substrate_state: tuple[int],
        system_ixs: tuple[int],
    ):
        # NOTE: add checks to make sure all indices are represented etc.

        # all indices inputing to the substrate units
        input_ixs = set([index for unit in self.units for index in unit.inputs])
        # prepare to marginalize out any input from units not present in the substrate
        marginalize_ixs = input_ixs - set(substrate_ixs)
        # prepare to condition unit tpms on inputs from outside the system
        condition_ixs = input_ixs - set(system_ixs) - marginalize_ixs

        substrate_units = []
        for u in tqdm(self.units, desc="Updating units"):

            if u.index in system_ixs:

                # copy unit to avoid destroying
                unit = deepcopy(u)

                self.update_unit_state(unit, substrate_ixs, substrate_state)

                # create tpm object
                tpm = pyphi.tpm.ExplicitTPM(unit.tpm)

                tpm = self.get_unit_tpm(
                    unit,
                    tpm,
                    substrate_ixs,
                    marginalize_ixs,
                    condition_ixs,
                    substrate_state,
                )

                # remaining "live" inputs
                live_inputs = tuple(
                    [
                        system_ixs.index(i)
                        for i in unit.inputs
                        if not i in marginalize_ixs.union(condition_ixs)
                    ]
                )

                # update state_dependent_tpms
                long_keys = []
                if type(unit.state_dependent_tpm) == dict:
                    new_state_dependent_tpm = dict()
                    for sub_state in pyphi.utils.all_states(len(substrate_ixs)):
                        long_state = tuple(
                            [
                                (
                                    0
                                    if not i in unit.inputs + (unit.index,)
                                    else sub_state[substrate_ixs.index(i)]
                                )
                                for i in range(len(u.substrate_state))
                            ]
                        )
                        new_tpm = pyphi.tpm.ExplicitTPM(
                            unit.state_dependent_tpm[long_state]
                        )
                        new_tpm = self.get_unit_tpm(
                            unit,
                            new_tpm,
                            substrate_ixs,
                            marginalize_ixs,
                            condition_ixs,
                            substrate_state,
                        )
                        new_state_dependent_tpm[sub_state] = new_tpm

                # get the updated unit
                substrate_unit = Unit(
                    system_ixs.index(unit.index),
                    live_inputs,
                    params=tpm,
                    tpm=tpm,
                    label=unit.label,
                    state=unit.state,
                    state_dependent_tpm=True,
                    inherited_tpm=new_state_dependent_tpm,
                )
                substrate_units.append(substrate_unit)

        return substrate_units

    def update_unit_state(
        self, unit: Unit, substrate_ixs: tuple[int], substrate_state: tuple[int]
    ):

        # check which substrate and state is stored in the unit
        old_substrate_state = self.state
        old_substrate_indices = self.node_indices

        # update substrate state with the state from kwargs
        new_substrate_state = tuple(
            [
                substrate_state[substrate_ixs.index(i)] if i in substrate_ixs else s
                for s, i in zip(old_substrate_state, old_substrate_indices)
            ]
        )

        # recreate unit with correct substrate state
        unit.set_substrate_state(new_substrate_state)

    def get_unit_tpm(
        self, unit, tpm, substrate_ixs, marginalize_ixs, condition_ixs, substrate_state
    ):

        # marginalize out non-substrate units from inputs
        non_substrate_inputs = [
            i for i, ix in enumerate(unit.inputs) if ix in marginalize_ixs
        ]
        tpm = tpm.marginalize_out(non_substrate_inputs)

        # condition on non-system units in inputs
        non_system_input_mapping = {
            unit.inputs.index(ix): s
            for ix, s in zip(substrate_ixs, substrate_state)
            if ix in condition_ixs and ix in unit.inputs
        }
        tpm = tpm.condition_tpm(non_system_input_mapping)

        # get tpm of only "live" inputs
        tpm = np.squeeze(tpm.tpm)[..., np.newaxis]

        return tpm

    def pyphi_kwargs(self, units: list[Unit]):

        unit_tpms = [
            np.concatenate((1 - unit.tpm, unit.tpm), axis=len(unit.tpm.shape) - 1)
            for unit in units
        ]
        node_labels = tuple([unit.label for unit in units])
        cm = np.array(
            [
                [1 if i in unit.inputs else 0 for i in range(len(units))]
                for unit in units
            ]
        )

        return dict(tpms=unit_tpms, node_labels=node_labels, cm=cm)

    def get_pyphi_kwargs(
        self,
        units: list[Unit] = None,
        substrate_indices: tuple[int] = None,
        substrate_state: tuple[int] = None,
        system_indices: tuple[int] = None,
    ):
        if units == None:
            units = self.units
        if substrate_indices == None:
            substrate_indices = tuple([unit.index for unit in units])
        if substrate_state == None:
            substrate_state = tuple([unit.state[0] for unit in units])
        if system_indices == None:
            system_indices = substrate_indices

        return self.pyphi_kwargs(
            self.subsystem_units(substrate_indices, substrate_state, system_indices)
        )
