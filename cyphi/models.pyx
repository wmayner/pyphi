#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

# NOTE: ``cimport`` is used to import special compile-time
# information about the numpy module (this is stored in a file numpy.pxd which
# is currently part of the Cython distribution, not the actualy numpy module
# itself; hence the strange almost-duplicate lines below.
import numpy as np
cimport numpy as np

# Fix a datatype for numpy arrays.
DTYPE = np.int
# ``ctypedef`` assigns a corresponding compile-time type to DTYPE_t. For every
# type in the numpy module there's a corresponding compile-time type with a
# _t-suffix.
ctypedef np.int_t DTYPE_t


cdef class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.

    :param connectivity_matrix: the network's connectivity matrix
    :type connectivity_matrix: ``np.ndarray``
    :param tpm: the network's transition probability matrix
    :type tpm: ``np.ndarray``
    :returns: a Network described by the given ``connectivity_matrix`` and
        ``tpm``

    """

    # Attributes of Cython "extension class" (classes defined with ``cdef``)
    # are defined outside of the constructor (the ``__cinit__`` function),
    # unlike regular Python
    cdef public np.ndarray connectivity_matrix, tpm

    def __init__(self, connectivity_matrix, tpm):
        self.connectivity_matrix = connectivity_matrix
        self.tpm = tpm
