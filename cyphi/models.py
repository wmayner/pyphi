# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

# TODO use structured arrays to represent sets of nodes, so labels
# are explicit rather than implicit (indices)?

import numpy as np
from . import utils
from .exceptions import ValidationException


class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.

    """
    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm):
        """
        Generates and initializes a set of nodes based on a connectivity matrix
        and a transition probability matrix.

        :param tpm: The network's transition probability matrix **in
            state-by-node form**
        :type tpm: ``np.ndarray``
        """

        # Validate the TPM
        if (tpm.ndim is not 2):
            raise ValidationException(
                "Invalid TPM: A Network's TPM must be 2-dimensional; " +
                "this TPM is " + str(tpm.ndim) + "-dimensional.")
        # TODO extend this hard-coded value to handle more than binary nodes
        if (tpm.size is not (2 ** tpm.shape[1]) * tpm.shape[1]):
            raise ValidationException(
                "Invalid TPM: There must be [ 2^(number of nodes) * " +
                "(number of nodes) ] elements in the TPM.")

        self.tpm = tpm
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by second dimension)
        self.size = tpm.shape[1]
        # Generate the the powerset of the node indices
        self.powerset = utils.powerset(np.arange(self.size))
        # Generate the nodes
        self.nodes = [Node(self, node_index)
                      for node_index in range(self.size)]
        # TODO handle nonbinary nodes
        self.num_states = 2 ** self.size
        self.uniform_distribution = np.divide(
            np.ones(self.num_states),
            self.num_states)

    def __repr__(self):
        return "Network(" + repr(self.tpm) + ")"

    def __str__(self):
        return "Network with TPM:\n" + str(self.tpm)

    def __eq__(self, other):
        """
        Two Networks are equal if they have the same TPM.
        """
        return np.array_equal(self.tpm, other.tpm)

    def __ne__(self, other):
        return not self.__eq__(other)



# TODO implement
class Mechanism(object):

    def __init__(self, network, nodes, state, MIP=None):
        # The network this mechanism is a part of
        self.network = network
        # The nodes in the mechanism
        self.nodes = nodes
        # The initial state of the mechanism
        self.state = state
        # The minimum information partition
        self.MIP = MIP

    # TODO calculate unconstrained repertoires here, or in cyphi.compute?

    pass


# TODO implement
class Distribution(object):
    """Probability distribution.
    """

    def __init__(self, data):
        """
        Data is a numpy array that should sum to one.
        """
        # Ensure ``data`` represents a probability distribution
        if np.sum(data) is not 1.0:
            raise ValidationException("Probabilities in a distribution must sum to 1.")
        self.data = data

