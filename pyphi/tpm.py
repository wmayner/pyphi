#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

"""
Provides the TPM, ExplicitTPM, and ImplicitTPM classes.
"""

from itertools import chain

import numpy as np

from . import config, convert, exceptions
from .utils import all_states, np_hash, np_immutable
from .constants import OFF, ON


class TPM:
    """A transition probability matrix."""

    def __init__(self):
        raise NotImplementedError

    @property
    def tpm(self):
        """Return the underlying `tpm` object."""
        return self._tpm

    def conditionally_independent(self):
        """Validate that the TPM is conditionally independent."""
        tpm = self._tpm
        tpm = np.array(tpm)
        if self.is_state_by_state():
            there_and_back_again = convert.state_by_node2state_by_state(
                convert.state_by_state2state_by_node(tpm)
            )
        else:
            there_and_back_again = convert.state_by_state2state_by_node(
                convert.state_by_node2state_by_state(tpm)
            )
        if not np.allclose((tpm - there_and_back_again), 0.0):
            raise exceptions.ConditionallyDependentError(
                "TPM is not conditionally independent.\n"
                "See the conditional independence example in the documentation "
                "for more info."
            )
        return True

    def validate(self):
        """Ensure the tpm is well-formed."""
        raise NotImplementedError

    def condition_tpm(self, fixed_nodes, state):
        """Return a TPM conditioned on the given fixed node indices, whose
        states are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable
        for broadcasting. The number of dimensions of the conditioned TPM will
        be the same as the unconditioned TPM.

        Args:
            fixed_nodes (Iterable[int]): The nodes the TPM will be conditioned on.
            state (Iterable[int]): The state of the fixed nodes.

        Returns:
            TPM: A conditioned TPM with the same number of dimensions,
            with singleton dimensions for nodes in a fixed state.
        """
        # Assumes multidimensional form
        conditioning_indices = [[slice(None)]] * (self.ndim - 1)
        for i, state_i in zip(fixed_nodes, state):
            # Preserve singleton dimensions with `np.newaxis`
            # TODO use utils.state_of and refactor nonvirtualized effect
            # repertoire to use this
            conditioning_indices[i] = [state_i, np.newaxis]
        # Flatten the indices.
        conditioning_indices = tuple(chain.from_iterable(conditioning_indices))
        # Obtain the actual conditioned TPM by indexing with the conditioning
        # indices.
        tpm = self._tpm[conditioning_indices]
        # Create new TPM object of the same type as self.
        # self.tpm has already been validated and converted to multidimensional
        # state-by-node form. Further validation would be problematic for
        # singleton dimensions.
        return type(self)(tpm, validate=False)

    def marginalize_out(self, node_indices):
        """Marginalize out nodes from this TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.

        Returns:
            ExplicitTPM: A TPM with the same number of dimensions, with the nodes
            marginalized out.
        """
        tpm = self._tpm.sum(tuple(node_indices), keepdims=True) / (
            np.array(self.shape)[list(node_indices)].prod()
        )
        # Return new TPM object of the same type as self.
        # self._tpm has already been validated and converted to multidimensional
        # state-by-node form. Further validation would be problematic for
        # singleton dimensions.
        return type(self)(tpm, validate=False)

    def is_deterministic(self):
        """Return whether the TPM is deterministic."""
        return np.all(np.logical_or(self._tpm == 1, self._tpm == 0))

    def is_state_by_state(self):
        """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
        ``False``.
        """
        return self.ndim == 2 and self.shape[0] == self.shape[1]

    def subtpm(self, fixed_nodes, state):
        """Return the TPM for a subset of nodes, conditioned on other nodes.

        Arguments:
            fixed_nodes (tuple[int]): The nodes to select.
            state (tuple[int]): The state of the fixed nodes.

        Returns:
            ExplicitTPM: The TPM of just the subsystem of the free nodes.

        Examples:
            >>> from pyphi import examples
            >>> # Get the TPM for nodes only 1 and 2, conditioned on node 0 = OFF
            >>> subtpm(examples.grid3_network().tpm.tpm, (0,), (0,))
            ExplicitTPM(array([[[[0.02931223, 0.04742587],
                                 [0.07585818, 0.88079708]],
            <BLANKLINE>
                                [[0.81757448, 0.11920292],
                                 [0.92414182, 0.95257413]]]]))
        """
        N = self._tpm.shape[-1]
        free_nodes = sorted(set(range(N)) - set(fixed_nodes))
        conditioned = self.condition_tpm(fixed_nodes, state)
        # TODO test indicing behavior on xr.DataArray
        return conditioned[..., free_nodes]

    def expand_tpm(self):
        """Broadcast a state-by-node TPM so that singleton dimensions are expanded
        over the full network.
        """
        unconstrained = np.ones([2] * (self._tpm.ndim - 1) + [self._tpm.shape[-1]])
        return type(self)(self._tpm * unconstrained, validate=False)

    def infer_edge(self, a, b, contexts):
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We
        call the state of |A'| the context |C| of |A|. There is an edge from |A|
        to |B| if there exists any context |C(A)| such that
        |Pr(B | C(A), A=0) != Pr(B | C(A), A=1)|.

        Args:
            a (int): The index of the putative source node.
            b (int): The index of the putative sink node.
            contexts (tuple[tuple[int]]): The tuple of states of ``a``
        Returns:
            bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
        """

        def a_in_context(context):
            """Given a context C(A), return the states of the full system with A
            OFF and ON, respectively.
            """
            a_off = context[:a] + OFF + context[a:]
            a_on = context[:a] + ON + context[a:]
            return (a_off, a_on)

        def a_affects_b_in_context(tpm, context):
            """Return ``True`` if A has an effect on B, given a context."""
            a_off, a_on = a_in_context(context)
            return tpm[a_off][b] != tpm[a_on][b]

        tpm = self.to_multidimensional_state_by_node()
        return any(a_affects_b_in_context(tpm, context) for context in contexts)

    def infer_cm(self):
        """Infer the connectivity matrix associated with a state-by-node TPM in
        multidimensional form.
        """
        tpm = self.to_multidimensional_state_by_node()
        network_size = tpm.shape[-1]
        all_contexts = tuple(all_states(network_size - 1))
        cm = np.empty((network_size, network_size), dtype=int)
        for a, b in np.ndindex(cm.shape):
            cm[a][b] = self.infer_edge(a, b, all_contexts)
        return cm

    def tpm_indices(self):
        """Return the indices of nodes in the TPM."""
        # TODO This currently assumes binary elements (2)
        return tuple(np.where(np.array(self.shape[:-1]) == 2)[0])

    def print(self):
        tpm = convert.to_multidimensional(self._tpm)
        for state in all_states(tpm.shape[-1]):
            print(f"{state}: {tpm[state]}")

    def __getattr__(self, name):
        if not "_tpm" in vars(self):
            raise AttributeError
        return getattr(self._tpm, name)

    def __getitem__(self, i):
        item = self._tpm[i]
        if isinstance(item, type(self._tpm)):
            item = type(self)(item, validate=False)
        return item

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()


class ExplicitTPM(TPM):
    """An explicit network TPM in multidimensional form."""

    def __init__(self, tpm, validate=True):
        self._tpm = np.array(tpm)

        if validate:
            self.validate(check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE)
            self._tpm = self.to_multidimensional_state_by_node()

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    def _validate_shape(self, check_independence=True):
        """Validate this TPM's shape.

        The TPM can be in

            * 2-dimensional state-by-state form,
            * 2-dimensional state-by-node form, or
            * multidimensional state-by-node form.
        """
        see_tpm_docs = (
            "See the documentation on TPM conventions and the `pyphi.Network` "
            "object for more information on TPM forms."
        )
        tpm = self._tpm
        # Get the number of nodes from the state-by-node TPM.
        N = tpm.shape[-1]
        if tpm.ndim == 2:
            if not (
                (tpm.shape[0] == 2 ** N and tpm.shape[1] == N)
                or (tpm.shape[0] == tpm.shape[1])
            ):
                raise ValueError(
                    "Invalid shape for 2-D TPM: {}\nFor a state-by-node TPM, "
                    "there must be "
                    "2^N rows and N columns, where N is the "
                    "number of nodes. State-by-state TPM must be square. "
                    "{}".format(tpm.shape, see_tpm_docs)
                )
            if tpm.shape[0] == tpm.shape[1] and check_independence:
                self.conditionally_independent()
        elif tpm.ndim == (N + 1):
            if tpm.shape != tuple([2] * N + [N]):
                raise ValueError(
                    "Invalid shape for multidimensional state-by-node TPM: {}\n"
                    "The shape should be {} for {} nodes. {}".format(
                        tpm.shape, ([2] * N) + [N], N, see_tpm_docs
                    )
                )
        else:
            raise ValueError(
                "Invalid TPM: Must be either 2-dimensional or multidimensional. "
                "{}".format(see_tpm_docs)
            )
        return True

    def _validate_probabilities(self):
        """Check that the probabilities in a TPM are valid."""
        if (self._tpm < 0.0).any() or (self._tpm > 1.0).any():
            raise ValueError(
                "Invalid TPM: probabilities must be in the interval [0, 1]."
            )
        if self.is_state_by_state() and np.any(np.sum(self._tpm, axis=1) != 1.0):
            raise ValueError("Invalid TPM: probabilities must sum to 1.")
        return True

    def validate(self, check_independence=True):
        """Validate this TPM."""
        return self._validate_probabilities() and self._validate_shape(
            check_independence
        )

    def to_multidimensional_state_by_node(self):
        """Return the current TPM re-represented in multidimensional
        state-by-node form.

        See the PyPhi documentation on :ref:`tpm-conventions` for more
        information.

        Returns:
            np.ndarray: The TPM in multidimensional state-by-node format.
        """
        if self.is_state_by_state():
            tpm = convert.state_by_state2state_by_node(self._tpm)
        else:
            tpm = convert.to_multidimensional(self._tpm)

        return tpm

    def squeeze(self, **kwargs):
        """Remove axes of length one from the TPM."""
        return type(self)(self._tpm.squeeze(**kwargs), validate=False)

    def __eq__(self, o: object):
        """Return whether this TPM equals the other object.

        Two TPMs are equal if they are instances of the ExplicitTPM class
        and their numpy arrays are equal.
        """
        return isinstance(o, ExplicitTPM) and np.array_equal(self._tpm, o._tpm)

    def __ne__(self, o: object):
        """Return whether this TPM is different from the other object.

        Two TPMs are equal if they are instances of the ExplicitTPM class
        and their numpy arrays are equal. Otherwise they are different.
        """
        return not self.__eq__(o)

    def __mul__(self, o):
        return type(self)(self._tpm * o._tpm, validate=False)

    def __repr__(self):
        return "ExplicitTPM({})".format(self._tpm)

    def __hash__(self):
        return self._hash


def reconstitute_tpm(subsystem):
    """Reconstitute the TPM of a subsystem using the individual node TPMs."""
    # The last axis of the node TPMs correponds to ON or OFF probabilities
    # (used in the conditioning step when calculating the repertoires); we want
    # ON probabilities.
    node_tpms = [node.tpm.tpm[..., 1] for node in subsystem.nodes]
    # Remove the singleton dimensions corresponding to external nodes
    node_tpms = [tpm.squeeze(axis=subsystem.external_indices) for tpm in node_tpms]
    # We add a new singleton axis at the end so that we can use
    # pyphi.tpm.expand_tpm, which expects a state-by-node TPM (where the last
    # axis corresponds to nodes.)
    node_tpms = [np.expand_dims(tpm, -1) for tpm in node_tpms]
    # Now we expand the node TPMs to the full state space, so we can combine
    # them all (this uses the maximum entropy distribution).
    node_tpms = [
        tpm * np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]]) for tpm in node_tpms
    ]
    # We concatenate the node TPMs along a new axis to get a multidimensional
    # state-by-node TPM (where the last axis corresponds to nodes).
    return np.concatenate(node_tpms, axis=-1)
