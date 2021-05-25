#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __tpm.py

"""
TPM classes.

The standard TPM class represents a State-by-State matrix where each row defines the probability 
the row's state will transition to any given column state. Xarray was chosen for its ability to define
each node as a dimension, easily accessible by its label and marginalizable.

The subclass of <snTPM> represents a State-by-Node matrix, only usable for binary systems, where each row
represent's the probability of each column Node being ON during the timestep after the given row's state.
"""

import xarray as xr
from pandas import DataFrame
import numpy as np

# labels provided?

# type checking: Union[ Iterable[Node]? ]
# mechanisms and purviews: Union[
#    Set[Node]
#    Iterable[Node]
# ]
# Node = Union[int, str]?

# Set[]


# Pass method calls to xarray?

# Should use only one underlying structure for TPM, but subclass when it's a
# matter of space or time
# e.g. subclass for state-by-node


class TPM:
    def __init__(self, tpm, previous_nodes, next_nodes=None, num_states_per_node=None):

        # If only one list of nodes is given, assumed to be a symmetric TPM.
        if next_nodes is None:
            next_nodes = previous_nodes.copy()

        # If a list of number of states is not given, assumed to be a binary TPM. 
        if num_states_per_node is None:
            num_states_per_node = [2 for x in previous_nodes + next_nodes]

        # Previous nodes are labelled with _p and next nodes with _n
        # to differentiate between nodes with the same name but different timestep
        dims = ["{}_p".format(node) for node in previous_nodes] + [
                "{}_n".format(node) for node in next_nodes]
        
        # The given numpy array is reshaped to have the appropriate number of dimensions for labeling
        self.tpm = xr.DataArray(tpm.reshape(num_states_per_node), dims=dims)

        if isinstance(tpm, DataFrame):
            num_states_per_node = [len(node_states) for node_states in tpm.index.levels + tpm.columns.levels]
            dims = ["{}_p".format(node) for node in tpm.index.names] + [
                    "{}_n".format(node) for node in tpm.columns.names]

            self.tpm = xr.DataArray(tpm.values.reshape(num_states_per_node), dims=dims)

    def tpm_indices(tpm):
        """Return the indices of nodes in the TPM."""
        raise NotImplementedError

    def is_deterministic(self) -> bool:
        """Return whether the TPM is deterministic."""
        raise NotImplementedError

    # Could get overridden by State-by-Node TPM's subclass to false? Or unneeded
    def is_state_by_state(self) -> bool:
        return True

    #def condition(self, fixed_nodes: Iterable, state) -> TPM:
        """Return a TPM conditioned on the given fixed node indices, whose states
        are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable for
        broadcasting. The number of dimensions of the conditioned TPM will be the
        same as the unconditioned TPM.
        """
    #    raise NotImplementedError

    def marginalize_out(self, node_indices):
        """Marginalize out nodes from a TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.
            tpm (np.ndarray): The TPM to marginalize the node out of.

        Returns:
            np.ndarray: A TPM with the same number of dimensions, with the nodes
            marginalized out.
        """
        raise NotImplementedError

    def infer_edge(tpm, a, b, contexts) -> bool:
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We call
        the state of |A'| the context |C| of |A|. There is an edge from |A| to |B|
        if there exists any context |C(A)| such that |Pr(B | C(A), A=0) != Pr(B |
        C(A), A=1)|.

        Args:
            tpm (np.ndarray): The TPM in state-by-node, multidimensional form.
            a (int): The index of the putative source node.
            b (int): The index of the putative sink node.

        Returns:
            bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
        """
        raise NotImplementedError

    def infer_cm(tpm):
        """Infer the connectivity matrix associated with a state-by-node TPM in
        multidimensional form.
        """
        raise NotImplementedError

    # TODO maybe not needed?
    def expand_tpm(tpm):
        """Broadcast a state-by-node TPM so that singleton dimensions are expanded
        over the full network.
        """
        raise NotImplementedError

    @classmethod
    def from_numpy():
        pass

    @classmethod
    def from_xarray():
        pass

    @classmethod
    def from_pands():
        pass
