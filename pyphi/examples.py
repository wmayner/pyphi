#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage and Examples
~~~~~~~~~~~~~~~~~~

The :class:`pyphi.network` object is the main object on which computations are
performed. It represents the network of interest.

The :class:`pyphi.subsystem` object is the secondary object; it represents a
subsystem of a network. |big_phi| is defined on subsystems.

The :mod:`pyphi.compute` module is the main entry-point for the library. It
contains methods for calculating concepts, constellations, complexes, etc. See
its documentation for details.

-------------------------------------------------------------------------------

Let's make a simple 3-node network and compute its |big_phi|.

To make a network, we need a TPM, current state, past state, and optionally a
connectivity matrix. The TPM can be in more than one form; see the
documentation for :class:`pyphi.network`. Here we'll use the 2-dimensional
state-by-node form.

    >>> import pyphi
    >>> import numpy as np
    >>> tpm = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0],
    ...                 [1, 1, 1], [1, 1, 1], [1, 1, 0]])

The current and past states should be |n|-tuples, where |n| is the number of
nodes in the network, where the |ith| element is the state of the |ith| node in
the network.

    >>> current_state = (1, 0, 0)
    >>> past_state = (1, 1, 0)

The connectivity matrix is a square matrix such that the |i,jth| entry is 1 if
there is a connection from node |i| to node |j|, and 0 otherwise.

    >>> cm = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]])

Now we construct the network itself with the arguments we just created:

    >>> network = pyphi.Network(tpm, current_state, past_state,
    ...                         connectivity_matrix=cm)

The next step is to define a subsystem for which we want to evaluate |big_phi|.
To make a subsystem, we need the indices of subset of nodes which should be
included in it and the network that the subsystem belongs to.

In this case, we want the |big_phi| of the entire network, so we simply include
every node in the network in our subsystem:

    >>> subsystem = pyphi.Subsystem(range(network.size), network)

Now we use :func:`pyphi.compute.big_phi` function to compute the |big_phi| of
our subsystem:

    >>> phi = pyphi.compute.big_phi(subsystem)
    >>> round(phi, pyphi.constants.PRECISION)
    2.312498

If we want to take a deeper look at the integrated-information-theoretic
properties of our network, we can access all the intermediate quantities and
structures that are caclulated in the course of arriving at a final |big_phi|
value by using :func:`pyphi.compute.big_mip`. This returns a deeply nested
object, |BigMip|, that contains data about the subsystem's constellation of
concepts, cause and effect repertoires, etc.

    >>> mip = pyphi.compute.big_mip(subsystem)

For instance, we can see that this network has 4 concepts:

    >>> len(mip.unpartitioned_constellation)
    4

The documentation for :mod:`pyphi.models` contains description of these
structures.

.. note::

    The network and subsystem discussed here are returned by the
    :func:`pyphi.examples.network` and :func:`pyphi.examples.subsystem`
    functions.
"""

import numpy as np
from .network import Network
from .subsystem import Subsystem


def network():
    """A simple 3-node network with roughly two bits of |big_phi|.

    Diagram::

               +~~~~~~~~+
          +~~~~|   A    |<~~~~+
          |    |  (OR)  +~~~+ |
          |    +~~~~~~~~+   | |
          |                 | |
          |                 v |
        +~+~~~~~~+      +~~~~~+~+
        |   B    |<~~~~~+   C   |
        | (COPY) +~~~~~>| (XOR) |
        +~~~~~~~~+      +~~~~~~~+

    TPM:

    +--------------+---------------+
    | Past state   | Current state |
    +--------------+---------------+
    |   C, B, A    |    A, B, C    |
    +==============+===============+
    |  {0, 0, 0}   |   {0, 0, 0}   |
    +--------------+---------------+
    |  {0, 0, 1}   |   {0, 0, 1}   |
    +--------------+---------------+
    |  {0, 1, 0}   |   {1, 0, 1}   |
    +--------------+---------------+
    |  {0, 1, 1}   |   {1, 0, 0}   |
    +--------------+---------------+
    |  {1, 0, 0}   |   {1, 1, 0}   |
    +--------------+---------------+
    |  {1, 0, 1}   |   {1, 1, 1}   |
    +--------------+---------------+
    |  {1, 1, 0}   |   {1, 1, 1}   |
    +--------------+---------------+
    |  {1, 1, 1}   |   {1, 1, 0}   |
    +--------------+---------------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 0 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+

    .. note::

        |CM[i][j] = 1| means that node |i| is connected to node |j|.
    """
    current_state = (1, 0, 0)
    past_state = (1, 1, 0)
    tpm = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]])

    cm = np.array([[0, 0, 1],
                   [1, 0, 1],
                   [1, 1, 0]])

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def subsystem():
    """A subsystem containing all the nodes of the
    :func:`pyphi.examples.network`."""
    net = network()
    return Subsystem(range(net.size), net)
