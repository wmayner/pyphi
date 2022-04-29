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

from pyphi.connectivity import get_inputs_from_cm
from pyphi.distribution import repertoire_shape

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
            p_states = [len(node_states) for node_states in tpm.index.levels]
            n_states = [len(node_states) for node_states in tpm.columns.levels]
            p_nodes = tpm.index.names
            n_nodes = tpm.columns.names
            dims = ["{}_p".format(node) for node in p_nodes] + [
                    "{}_n".format(node) for node in n_nodes]
            coords = dict(zip(dims, [np.array(range(state)) for state in p_states + n_states]))
            self.tpm = xr.DataArray(tpm.values.reshape(p_states + n_states, order="F"), coords=coords, dims=dims)
            

        else:    
            
            if isinstance(tpm, xr.DataArray):
                tpm = tpm.data

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
            coords = dict(zip(dims, [np.array(range(state)) for state in p_states + n_states]))
            self.tpm = xr.DataArray(tpm.reshape(p_states + n_states, order="F"), coords=coords, dims=dims)
        
        self.symmetric = bool(p_nodes == n_nodes)
        self._p_nodes = p_nodes
        self._n_nodes = n_nodes

        # So far only used in infer_cm, might be possible to rework that and remove this
        # Could also just be a list (or implicit in sum of p_nodes and n_nodes), states could
        # be accessed by tpm.shape[[p_nodes + n_nodes].index(node)]
        self.all_nodes = dict(zip(self.p_nodes + self.n_nodes, self.tpm.shape))

    @property
    def p_nodes(self):
        """Returns p_nodes formatted"""
        return ["{}_p".format(node) for node in self._p_nodes]
    
    @property
    def n_nodes(self):
        """Returns n_nodes formatted"""
        return ["{}_n".format(node) for node in self._n_nodes]

    @property
    def p_states(self):
        """Returns list of p_states"""
        return [self.all_nodes[p_node] for p_node in self.p_nodes]

    @property
    def n_states(self):
        """Returns list of n_states"""
        return [self.all_nodes[n_node] for n_node in self.n_nodes]

    # Maybe make this more generalized? Could be based on xarray's shape?
    # Maybe make it just one tuple? Hard to separate then...
    @property
    def tpm_indices(self):
        """Returns two tuples of indices for p_nodes and n_nodes"""
        return tuple(range(len(self.p_nodes))), tuple(range(len(self.n_nodes)))

        # return tuple(range(len(self.tpm.shape)))
        # return tuple(range(len(self.p_nodes)) + range(len(self.n_nodes))))

    @property
    def is_deterministic(self) -> bool:
        """Return whether the TPM is deterministic."""
        return np.all(np.logical_or(self.tpm.data == 0, self.tpm.data == 1))

    # Could get overridden by State-by-Node TPM's subclass to false? Or unneeded
    @property
    def is_state_by_state(self):
        return True

    @property
    def num_states(self):
        """int: Number of possible states the set of previous nodes can be in
        """
        # Number of states equal to product of states of p_nodes?
        # This doesn't really make sense in asymmetric TPMs, so maybe a different property
        # Makes more sense here? Or could split to num_p_states and num_n_states
        num_states = 1
        for node in self.p_nodes:
            num_states *= self.all_nodes[node]
        return num_states

    @property
    def num_n_states(self):
        """int: Number of possible states the set of next nodes can be in
        """
        num_states = 1
        for node in self.n_nodes:
            num_states *= self.all_nodes[node]
        return num_states

    @property
    def shape(self):
        return tuple(self.tpm.shape)

    @property
    def p_node_indices(self):
        return tuple(range(len(self.p_nodes)))

    @property
    def n_node_indices(self):
        return tuple(range(len(self.n_nodes)))

    # TODO Maybe try using xarray's coordinates feature to keep dims and state info?
    # As opposed to trying to just use slices, keeping the coordinate of the dimension
    # equal to the conditioned state could make things easier down the line
    # TODO Consider splitting into condition rows and condition columns, then could mix
    # into conditioning for asymmetric tpms (or force square dimensionality with empty dims)
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

            for i in fixed_nodes:
                self.p_states[i] = 1 
                self.all_nodes[self.p_nodes[i]] = 1
                self.all_nodes[self.n_nodes[i]] = 1

            return conditioned_tpm
        
        else:
            print(self, "symmetric?:", self.symmetric)
            raise NotImplementedError

    def create_node_tpm(self, index, cm):
        """Create a new TPM object based on this one, except only describing the
        transitions of a single node in the network.

        Args:
            node (int): The index of the node whose TPM we wish to create
        """
        # Want to marginalize out all other column nodes, but then don't need to worry about normalization
        node = self.n_nodes[index]
        other_nodes = [label for label in self.n_nodes if label != node]
        node_tpm = self.tpm.sum(dim=other_nodes)

        node_tpm = TPM(tpm=node_tpm, p_nodes=self._p_nodes, n_nodes=[self._n_nodes[index]], p_states=self.p_states, n_states=[self.all_nodes[node]])

        return node_tpm
        #return node_tpm.marginalize_out(tuple(set(self.p_node_indices) - set(get_inputs_from_cm(index, cm))), rows=True)

    def condition_node_tpm(node_tpm, fixed_nodes, state, col=False):
        """Condition a node TPM object, a special case of asymmetric TPMs
        """
        fixed_node_labels = [node_tpm.p_nodes[node] for node in fixed_nodes]
        indexer = dict(zip(fixed_node_labels, [state[node] for node in fixed_nodes]))

        kept_dims = [node_tpm.tpm.dims.index(label) for label in fixed_node_labels]
        conditioned_tpm = node_tpm.tpm.loc[indexer]

        # Regrow dimensions where they got trimmed
        for j in range(len(kept_dims)):
            conditioned_tpm = conditioned_tpm.expand_dims(fixed_node_labels[j], axis=kept_dims[j])

        if col:
            conditioned_tpm = conditioned_tpm.sum(dim=conditioned_tpm.dims[-1], keepdims=True)
        
        return conditioned_tpm

    def condition_list(tpm_list, fixed_nodes, state):
        """Condition a list of Node TPMs.

        Args:
            tpm_list (list[TPM]): The Node TPMs to be conditioned
            fixed_nodes (tuple(int)): The node indicies that are now fixed
            state (tuple(int)): The state of the system when conditioning
        """
        # Step 0: No conditioning required if fixed_nodes empty
        if fixed_nodes is ():
            return tpm_list

        # Step 1: Drop rows that do not fit with the conditioned state, for all tpms
        for i in range(len(tpm_list)):
            # Replace TPM in list's tpm with dropped row tpm.
            tpm_list[i].tpm = TPM.condition_node_tpm(tpm_list[i], fixed_nodes, state)

        # Step 2: If a particular tpm describes the transitions of a fixed node, sum the column dimension together
        for i in fixed_nodes:
            # NOTE: Assumes Node TPM only has 1 column dimension. Can be generalized, but needs more information about labels.
            # Since we're already assuming a list, however, this is a valid assumption. See the more general condition method
            # For implementing generic asymmetric conditioning
            tpm_list[i].tpm = tpm_list[i].tpm.sum(dim=tpm_list[i].tpm.dims[-1], keepdims=True)

        return tpm_list
            

    # TODO Currently only works for symmetric TPMs
    # TODO **kwargs to determine if marginalizing out just row/column?
    def marginalize_out(self, node_indices, rows=False):
        """Marginalize out nodes from a TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.
            Index based on dimension of the node.
            rows (bool): Whether to marginalize on only the rows


        Returns:
            xarray: A tpm with the same number of dimensions, with the nodes
        marginalized out.
        """
        def normalize(tpm):
            """Returns a normalized TPM after marginalization"""
            return tpm / (np.array(self.tpm.shape)[list(node_indices)].prod())

        if self.symmetric:           
            labels = [self.p_nodes[i] for i in node_indices] + [self.n_nodes[i] for i in node_indices]
            new_p_states = self.p_states
            new_n_states = self.n_states
            for i in node_indices:
                new_p_states[i], new_n_states[i] = 1
                
        elif rows:
            labels = [self.p_nodes[i] for i in node_indices]
            new_p_states = self.p_states
            for i in node_indices:
                new_p_states[i] = 1
            new_n_states = self.n_states

        else:
            raise NotImplementedError

        marginalized_tpm = self.tpm

        for label in labels:
            marginalized_tpm = marginalized_tpm.sum(dim=label, keepdims=True)

        finished_tpm = normalize(marginalized_tpm)

        return TPM(tpm=finished_tpm, p_nodes=self._p_nodes, n_nodes=self._n_nodes, p_states=new_p_states, n_states=new_n_states)


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
            temp_n_nodes = tpm.n_nodes
            name = temp_n_nodes[b]
            # Instead of making a full copy, just remove and insert afterwards
            temp_n_nodes.remove(name)
            marginalized = tpm.tpm.groupby(name).sum(temp_n_nodes).loc[state]
            temp_n_nodes.insert(b, name)
            return marginalized

        def a_affects_b_in_context(context):
            """Return ``True`` if A has an effect on B, given a context."""
            a_states = a_in_context(context)
            comparator = marginalize_b(tuple(a_states[0])).data.round(12)
            return any(not np.array_equal(comparator, marginalize_b(state).data.round(12)) for state in a_states[1:])

        return any(a_affects_b_in_context(context) for context in contexts)

    # Takes TPM object, could use self instead?
    def infer_cm(tpm):
        """Infer the connectivity matrix associated with a state-by-state TPM in
        object form.
        """
        # Set up empty cm based on nodes
        cm = np.empty((len(tpm._p_nodes), len(tpm._n_nodes)), dtype=int)
        # Iterate through every node pair
        for a, b in np.ndindex(cm.shape):
            # Determine context states based on a
            a_prime = tpm.p_nodes
            a_prime.pop(a)
            contexts = tuple(product(*tuple(tuple(range(tpm.all_nodes[node])) for node in a_prime)))
            cm[a][b] = tpm.infer_edge(a, b, contexts)
        return cm

    def infer_node_edge(tpm, a, contexts):

        def a_in_context(context):
            """Given a context C(A), return the states of the full system with A
            in each of its possible states, in order as a list.
            """
            a_states = [
                context[:a] + (i, ) + context[a:]
                for i in range(tpm.tpm.shape[a])
            ]
            return a_states

        def a_affects_b_in_context(context):
            a_states = np.array(a_in_context(context))
            return any(not np.array_equal(tpm.tpm.data[tuple(a_states[0])], tpm.tpm.data[tuple(state)]) for state in a_states[1:])

        return any(a_affects_b_in_context(context) for context in contexts)

    def infer_node_cm(tpm):
        cm = np.empty((len(tpm._p_nodes), 1), dtype=int)
        
        for a, b in np.ndindex(cm.shape):
            # Determine context states based on a
            a_prime = tpm.p_nodes
            a_prime.pop(a)
            contexts = tuple(product(*tuple(tuple(range(tpm.all_nodes[node])) for node in a_prime)))
            cm[a][b] = tpm.infer_node_edge(a, contexts)

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

    def copy(self):
        p_states = [self.tpm.shape[self.tpm.dims.index(node)] for node in self.p_nodes]
        n_states = [self.tpm.shape[self.tpm.dims.index(node)] for node in self.n_nodes]
        return TPM(self.tpm, p_nodes=self._p_nodes, p_states=p_states, n_nodes=self._n_nodes, n_states=n_states)

    def sum(self, dim, keepdims=True):
        return self.tpm.sum(dim, keepdims)

    def __repr__(self):
        return self.tpm.__repr__()

# TODO Better name?
class SbN(TPM):
    """The subclass of <SbN> represents a State-by-Node matrix, only usable for binary 
    systems, where each row represents the probability of each column Node being ON 
    during the timestep after the given row's state.
    """
    def __init__(self, tpm, p_nodes=None, p_states=None, n_nodes=None, n_states=None):
            format = False

            if isinstance(tpm, xr.DataArray):
                tpm = tpm.data

            # Not multi-dimensional SbN
            if tpm.ndim == 2:
                if tpm.shape[0] == tpm.shape[1]: # SbS
                    format = True
                elif n_nodes: # If n_nodes isn't given, we have to assume SbN unless we ask the user to specify a flag
                    if len(n_nodes) != tpm.shape[1]: # SbS
                        format = True

            # If format is true, change from SbS to SbN
            if format:
                super().__init__(tpm, p_nodes, p_states, n_nodes, n_states)
                #bin_node_tpms = [self.tpm.sel({node: 1}).sum(self.n_nodes.copy().remove(node)).expand_dims("nodes", axis=-1) 
                #for node in self.n_nodes]
                bin_node_tpms=[]
                # TODO Is there a way to do this with some form of list comp? unfortunately using .copy()
                # seems to break things if I try list comp
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

                self._p_nodes = p_nodes
                # Differences: Shape is going to be (S_a, S_b, S_c... N), rows are like normal but index of last is the size of the n_nodes list
                dims = self.p_nodes + ["n_nodes"]
                
                # Binary only, so num_states_per_node is tuple of 2s with length of p_nodes
                p_states = [2] * len(p_nodes)

                # Need to keep track of location of nodes in the last dimension
                self._n_nodes = n_nodes
                self.tpm = xr.DataArray(tpm.reshape(p_states + [len(n_nodes)], order="F"), dims=dims)

            self.all_nodes = self.all_nodes = dict(zip(self.p_nodes + self.n_nodes, [2 for i in self.p_nodes + self.n_nodes]))

    # TODO Only valid for symmetric tpms
    def tpm_indices(self):
        """Return the indices of nodes in the SbN."""
        return tuple(np.where(np.array(self.tpm.shape[:-1]) == 2)[0])

    def is_state_by_state(self):
        return False

    def get_node_transitions(self, state, external):
        """Intended for node TPMs only; but viable for any.
        Returns the np.array of the transition data for a given state.
        """
        if not external:
            n_tpm = self.tpm
            indexer = dict({self.tpm.dims[-1]: state})
            return n_tpm.loc[indexer].data
        
        else:
            return self.tpm.data


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

    def copy(self):
        return SbN(self.tpm, p_nodes=self._p_nodes, n_nodes=self._n_nodes)

    def create_node_tpm(self, index, cm):
        # Grab the column which holds the transition data for the desired node
        indexer = dict({"n_nodes": index})
        node_tpm = self.tpm.loc[indexer]

        # Expand the dimension so that we can add more data
        node_tpm = node_tpm.expand_dims(dim='n0_n', axis=-1)

        # Pad the trimmed node_tpm, with NaN values with length 1 before
        # The current data (as current data is when the node is at state 1)
        mapping = dict({"n0_n": (1, 0)})
        node_tpm = node_tpm.pad(mapping)

        # Replace NaN data with data on when it will in state 0 (1 - state 1)
        node_tpm[..., 0] = 1 - node_tpm[..., 1]

        node_TPM = TPM(node_tpm, p_nodes=self._p_nodes).marginalize_out(tuple(set(self.p_node_indices) - set(get_inputs_from_cm(index, cm))), rows=True)
        # Create new TPM object with this data
        return node_TPM

    def __repr__(self):
        return 1
