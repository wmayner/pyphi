#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __tpm.py

"""
TPM classes.

TPM -> Works with all state-by-state tpms 
(binary, nonbinary, symmetric, asymmetric -> node TPMs)

SbN -> Only works with state-by-node tpms,
which are only binary. Can work with asymmetric tpms, however.
"""

import xarray as xr
from pandas import DataFrame
import numpy as np
from string import ascii_uppercase
from itertools import product
from math import log2

from .utils import all_states

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
    """General class of TPM objects in state-by-state form. 

    The standard TPM class represents a State-by-State matrix where each row defines the 
    probability that the row's state will transition to any given column state. 

    Xarray was chosen for its ability to define each node as a dimension, 
    easily accessible by its label and marginalizable.

    Can accept both DataFrame objects and 2D or pre-shaped multidimensional 
    numpy arrays.
    """
    def __init__(self, tpm, p_nodes=None, p_states=None, n_nodes=None, n_states=None):

        if isinstance(tpm, DataFrame):
            num_states_per_node = [len(node_states) for node_states in tpm.index.levels + tpm.columns.levels]
            p_nodes = tpm.index.names
            n_nodes = tpm.columns.names
            dims = ["{}_p".format(node) for node in p_nodes] + [
                    "{}_n".format(node) for node in n_nodes]

            self.tpm = xr.DataArray(tpm.values.reshape(num_states_per_node, order="F"), dims=dims)

        else:    
            
            if p_states is None: # Binary
                if p_nodes is None:
                    # Gen p_nodes
                    p_nodes = ["n{}".format(i) for i in range(int(log2(tpm.shape[0])))]

                if n_nodes is None: 
                    # NOTE Specifying a Node TPM with just data and p_nodes (and perhaps p_states and n_states if nb)
                    # seems really useful, so I want a label generation method, but this can give awkward results
                    # Gen n_nodes
                    n_nodes = ["n{}".format(i) for i in range(int(log2(tpm.shape[1])))]
                
                # Gen p_states
                p_states = [2] * len(p_nodes)
                # Gen n_states
                n_states = [2] * len(n_nodes)

            else: # Non-binary
                if p_nodes is None:
                    # Gen p_nodes
                    p_nodes = ["n{}".format(i) for i in range(len(p_states))]

                if n_nodes is None: # Nbin, Sym
                    if n_states is None:    
                        # Cpy p_states -> n_states
                        n_states = p_states.copy()
                        # Cpy p_nodes -> n_nodes
                        n_nodes = p_nodes.copy()
                    
                    else: # n_states specified so assumed different from p_states?
                        n_nodes = ["n{}".format(i) for i in range(len(n_states))]

                # Else Nbin, Asym, pass as should have all specified


            # Previous nodes are labelled with _p and next nodes with _n
            # to differentiate between nodes with the same name but different timestep
            dims = ["{}_p".format(node) for node in p_nodes] + [
                    "{}_n".format(node) for node in n_nodes]
            
            # The given numpy array is reshaped to have the appropriate number of dimensions for labeling
            # Fortran, or little-endian ordering is used, meaning left-most (first called) node varies fastest
            self.tpm = xr.DataArray(tpm.reshape(p_states + n_states, order="F"), dims=dims)
        
        self.symmetric = bool(p_nodes == n_nodes)
        self.p_nodes = ["{}_p".format(node) for node in p_nodes]
        self.n_nodes = ["{}_n".format(node) for node in n_nodes]

        # So far only used in infer_cm, might be possible to rework that and remove this
        self.all_nodes = dict(zip(self.p_nodes + self.n_nodes, self.tpm.shape))

    # Maybe make this more generalized? Could be based on xarray's shape?
    # Maybe make it just one tuple? Hard to separate then...
    def tpm_indices(self):
        """Returns two tuples of indices for p_nodes and n_nodes"""
        return tuple(range(len(self.p_nodes))), tuple(range(len(self.n_nodes)))

        # return tuple(range(len(self.tpm.shape)))
        # return tuple(range(len(self.p_nodes)) + range(len(self.n_nodes))))

    def is_deterministic(self) -> bool:
        """Return whether the TPM is deterministic."""
        return np.all(np.logical_or(self.tpm.data == 0, self.tpm.data == 1))

    # Could get overridden by State-by-Node TPM's subclass to false? Or unneeded
    def is_state_by_state(self):
        return True

    # TODO Maybe try using xarray's coordinates feature to keep dims and state info?
    # As opposed to trying to just use slices, keeping the coordinate of the dimension
    # equal to the conditioned state could make things easier down the line
    def condition(self, fixed_nodes, state):
        """Return a TPM conditioned on the given fixed node indices, whose states
        are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable for
        broadcasting. The number of dimensions of the conditioned TPM will be the
        same as the unconditioned TPM.

        Args:
            fixed_nodes: tuple of indicies of nodes that are fixed
        """
        # Only doing this for symmetric tpms at the moment
        if self.symmetric:
            fixed_node_labels = [self.p_nodes[node] for node in fixed_nodes]
            indexer = dict(zip(fixed_node_labels, [state[node] for node in fixed_nodes]))
            # Drop rows that don't fit with the conditioned nodes
            # TODO Better way to keep dims? IndexSlicer maybe?
            # Tried using None slices but dimensions still got removed :(
            kept_dims = [self.tpm.dims.index(label) for label in fixed_node_labels]
            conditioned_tpm = self.tpm.loc[indexer]

            # Regrow dimensions where they got trimmed
            for i in range(len(kept_dims)):
                conditioned_tpm = conditioned_tpm.expand_dims(fixed_node_labels[i], axis=kept_dims[i])

            # Marginalize across columns that don't fit with the conditioned nodes
            # Because assumed symmetry, self.n_nodes is same as self.p_nodes, except for labelling
            # Since we're summing across columns, need labels from n_nodes

            column_labels = [self.n_nodes[node] for node in fixed_nodes]

            for label in column_labels:
                conditioned_tpm = conditioned_tpm.sum(dim=label, keepdims=True)
            # At some point maybe change the dict of self.all_nodes? Perhaps ideally make all_nodes unneeeded

            return conditioned_tpm
        # TODO how much sense does it make to do this for asymmetric cases?
        else:
            raise NotImplementedError

    # TODO Currently only works for symmetric TPMs
    # TODO **kwargs to determine if marginalizing out just row/column?
    def marginalize_out(self, node_indices):
        """Marginalize out nodes from a TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.
            Index based on dimension of the node.
            self: The TPM object to marginalize the node(s) out of.

        Returns:
            xarray: A tpm with the same number of dimensions, with the nodes
        marginalized out.
        """
        def normalize(tpm):
            """Returns a normalized TPM after marginalization"""
            return tpm / (np.array(self.tpm.shape)[list(node_indices)].prod())

        if self.symmetric:
            
            labels = [self.p_nodes[i] for i in node_indices] + [self.n_nodes[i] for i in node_indices]

            marginalized_tpm = self.tpm

            for label in labels:
                marginalized_tpm = marginalized_tpm.sum(dim=label, keepdims=True)

            return normalize(marginalized_tpm)
        else:
            raise NotImplementedError

    def infer_edge(tpm, a, b, contexts):
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We call
        the state of |A'| the context |C| of |A|. There is an edge from |A| to |B|
        if there exists any context |C(A)| such that |Pr(B | C(A), A=0) != Pr(B |
        C(A), A=1)|.

        Args:
            tpm (np.ndarray): The TPM as an object
            a (int): The index of the putative source node.
            b (int): The index of the putative sink node.
        Returns:
            bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
        """
        def a_in_context(context):
            """Given a context C(A), return the states of the full system with A
            in each of its possible states, in order as a list.
            """
            a_states = [
                context[:a] + (i, ) + context[a:]
                for i in range(tpm.tpm.shape[a])
            ]
            return a_states

        def marginalize_b(state):
            """Return the distribution of possible states of b at t+1"""
            name = tpm.n_nodes[b]
            # Instead of making a full copy, just remove and insert afterwards
            tpm.n_nodes.remove(name)
            marginalized = tpm.tpm.groupby(name).sum(tpm.n_nodes).loc[tuple(state)]
            tpm.n_nodes.insert(b, name)
            return marginalized

        def a_affects_b_in_context(context):
            """Return ``True`` if A has an effect on B, given a context."""
            a_states = a_in_context(context)
            comparator = marginalize_b(a_states[0]).round(12)
            return any(not comparator.equals(marginalize_b(state).round(12)) for state in a_states[1:])

        return any(a_affects_b_in_context(context) for context in contexts)

    # Takes TPM object, could use self instead?
    def infer_cm(tpm):
        """Infer the connectivity matrix associated with a state-by-state TPM in
        object form.
        """
        # Set up empty cm based on nodes
        cm = np.empty((len(tpm.p_nodes), len(tpm.n_nodes)), dtype=int)
        # Iterate through every node pair
        for a, b in np.ndindex(cm.shape):
            # Determine context states based on a
            a_prime = tpm.p_nodes.copy()
            a_prime.pop(a)
            contexts = tuple(product(*tuple(tuple(range(tpm.all_nodes[node])) for node in a_prime)))
            cm[a][b] = tpm.infer_edge(a, b, contexts)
        return cm

    def __getitem__(self, key):
        return self.tpm[key]

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

# TODO Better name?
# TODO make n_nodes optional
# TODO make p_nodes optional 
class SbN(TPM):
    """The subclass of <SbN> represents a State-by-Node matrix, only usable for binary 
    systems, where each row represents the probability of each column Node being ON 
    during the timestep after the given row's state.
    """
    def __init__(self, tpm, p_nodes=None, p_states=None, n_nodes=None, format=False, n_states=None):

            # If format is true, change from SbS to SbN
            if format:
                super().__init__(tpm, p_nodes, p_states, n_nodes, n_states)
                #bin_node_tpms = [self.tpm.sel({node: 1}).sum(self.n_nodes.copy().remove(node)).expand_dims("nodes", axis=-1) 
                #for node in self.n_nodes]
                bin_node_tpms=[]
                # TODO Is there a way to do this with list comp? unfortunately using .copy()
                # seems to break things if I try
                # TODO Can we use the coordinates property to name the nodes in the "nodes" dimension?
                for node in self.n_nodes:
                    temp = self.n_nodes.copy()
                    temp.remove(node)
                    bin_node_tpms.append(self.tpm.sel({node: 1}).sum(temp).expand_dims("n_nodes", axis=-1))
                self.tpm = xr.concat(bin_node_tpms, dim="n_nodes")

            # If format is false, is already in SbN form
            else:     
                if p_nodes is None:
                    p_nodes = ["n{}".format(i) for i in range(int(log2(np.prod(tpm.shape[:-1]))))]

                # NOTE: This will produce a different naming scheme in some instances than using
                # the super constructor, not too big of a deal but worth noting 
                # Fixing it would probably require unnecessary convolution, only worthwhile
                # if it is actually an issue
                if n_nodes is None:
                    n_nodes = ["n{}".format(i) for i in range(tpm.shape[-1])]

                self.p_nodes = ["{}_p".format(node) for node in p_nodes]
                # Differences: Shape is going to be (S_a, S_b, S_c... N), rows are like normal but index of last is the size of the n_nodes list
                dims = self.p_nodes + ["n_nodes"]
                
                # Binary only, so num_states_per_node is tuple of 2s with length of p_nodes
                p_states = [2] * len(p_nodes)

                # Need to keep track of location of nodes in the last dimension
                self.n_nodes = ["{}_n".format(node) for node in n_nodes]
                self.tpm = xr.DataArray(tpm.reshape(p_states + [len(n_nodes)], order="F"), dims=dims)

    # TODO Only valid for symmetric tpms
    def tpm_indices(self):
        """Return the indices of nodes in the SbN."""
        return tuple(np.where(np.array(self.tpm.shape[:-1]) == 2)[0])

    def is_state_by_state(self):
        return False

    # SbN form
    def infer_edge(tpm, a, b, contexts):
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We call
        the state of |A'| the context |C| of |A|. There is an edge from |A| to |B|
        if there exists any context |C(A)| such that |Pr(B | C(A), A=0) != Pr(B |
        C(A), A=1)|.

        Args:
            tpm (SbN): The TPM in state-by-node, multidimensional form.
            a (int): The index of the putative source node.
            b (int): The index of the putative sink node.

        Returns:
            bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
        """
        def a_in_context(context):
            """Given a context C(A), return the states of the full system with A
            OFF (0) and ON (1), respectively.
            """
            a_off = context[:a] + (0, ) + context[a:]
            a_on = context[:a] + (1, ) + context[a:]
            return (a_off, a_on)

        def a_affects_b_in_context(context):
            """Return ``True`` if A has an effect on B, given a context."""
            a_off, a_on = a_in_context(context)
            return tpm.tpm[a_off][b] != tpm.tpm[a_on][b]

        return any(a_affects_b_in_context(context) for context in contexts)
    
    # SbN form
    def infer_cm(tpm):
        """Infer the connectivity matrix associated with a SbN tpm in
        multidimensional form.
        """
        network_size = tpm.tpm.shape[-1]
        all_contexts = tuple(all_states(network_size - 1))
        cm = np.empty((network_size, network_size), dtype=int)
        for a, b in np.ndindex(cm.shape):
            cm[a][b] = SbN.infer_edge(tpm, a, b, all_contexts)
        return cm


    def marginalize_out(self, node_indices):
        """Marginalize out nodes from a TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.
            tpm (np.ndarray): The TPM to marginalize the node out of.

        Returns:
            np.ndarray: An SbN with the same number of dimensions, with the nodes
            marginalized out.
        """
        def normalize(tpm):
            return tpm / (np.array(self.tpm.shape)[list(node_indices)].prod())

        marginalized_tpm = self.tpm 
    
        for label in [self.p_nodes[i] for i in node_indices]:
            marginalized_tpm = marginalized_tpm.sum(dim=label, keepdims=True)

        return normalize(marginalized_tpm)

    def condition(self, fixed_nodes, state):
        """Return a TPM conditioned on the given fixed node indices, whose states
        are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable for
        broadcasting. The number of dimensions of the conditioned TPM will be the
        same as the unconditioned TPM.
        """
        fixed_node_labels = [self.p_nodes[node] for node in fixed_nodes]
        indexer = dict(zip(fixed_node_labels, [state[node] for node in fixed_nodes]))
        kept_dims = [self.tpm.dims.index(label) for label in fixed_node_labels]
        # Throw out rows that don't fit with fixed node states, don't need to worry
        # about columns as they are already essentially marginalized
        conditioned_tpm = self.tpm.loc[indexer]

        # Regrow dimensions where they got trimmed
        for i in range(len(kept_dims)):
            conditioned_tpm = conditioned_tpm.expand_dims(dim=fixed_node_labels[i], axis=kept_dims[i])
        return conditioned_tpm
