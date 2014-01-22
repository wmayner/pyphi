# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

import numpy as np
from itertools import chain
from . import utils
from .exceptions import ValidationException

class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.
    """

    def __init__(self, connectivity_matrix, tpm):
        """
        :param connectivity_matrix: The network's connectivity matrix (must be
            square)
        :type connectivity_matrix: ``np.ndarray``
        :param tpm: The network's transition probability matrix
        :type tpm: ``np.ndarray``

        :returns: a Network described by the given ``connectivity_matrix`` and
            ``tpm``
        """

        # Ensure connectivity matrix is square
        if len(connectivity_matrix.shape) is not 2 or \
            connectivity_matrix.shape[0] is not connectivity_matrix.shape[1]:
            raise ValidationException("Connectivity matrix must be square.")

        self.connectivity_matrix = connectivity_matrix
        self.tpm = tpm

        # Generate powerset
        self.powerset = utils.powerset(np.arange(connectivity_matrix.shape[0]))
