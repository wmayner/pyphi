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


"""
Residue Example

-------------------------------------------------------------------------------

The residue example describes  a system containing two AND nodes A and B
with a single overlapping input node

First lets create the subsystem corresponding to the residue network

   >>> subsystem = residue_subsystem()

Next we can define the mechanisms of interest

   >>> A = subsystem.indices2nodes((0,))
   >>> B = subsystem.indices2nodes((1,))
   >>> AB = subsystem.indices2nodes((0,1))

and possible past purviews

   >>> CD = subsystem.indices2nodes((2,3))
   >>> DE = subsystem.indices2nodes((3,4))
   >>> CDE = subsystem.indices2nodes((2,3,4))

We can then evaluate the cause information for each of the mechanisms over the
past purview CDE,

   >>> subsystem.cause_info(A, CDE)
   0.33333191666400036

   >>> subsystem.cause_info(B, CDE)
   0.33333191666400036

   >>> subsystem.cause_info(AB, CDE)
   0.49999972500000006

The composite mechanism AB has greater cause information than either of the individual
mechanisms. This goes against the idea that AB should exist minimally in this system.

Instead, we can quantify existence as the irreducible cause information of a mechanism.
The MIP of a mechanism is the partition of mechanism and purview which makes the least
difference to the cause repertoire. The irreducible cause information is the distance
between the unpartitioned and partitioned repertoires.

To calculate the MIP structure of AB

   >>> mip_AB = subsystem.mip_past(AB, CDE)

We can then determine what the specific partition is

   >>> mip_AB.partition
   (Part(mechanism=(), purview=(n2,)), Part(mechanism=(n0, n1), purview=(n3, n4)))

The labels (n0, n1, n2, n3, n4) correspond to nodes (A,B,C,D,E) respectively. Thus the
MIP is (AB | DE) X ([] | C), where [] denotes the empty mechanism.

The partitioned repertoire of the MIP can also be retrieved,

   >>> mip_AB.partitioned_repertoire.flatten(order='F')
   array([ 0.2,  0.2,  0.1,  0.1,  0.2,  0.2,  0. ,  0. ])

and we can then calculate the irreducible cause information as the difference between
partitioned and unpartitioned repertoires,

   >>> mip_AB.phi
   0.09999990000000035

One counter-intuitive result which merits discussion is that since irreducible cause information
is what defines existence, we must also evaluate the irreducible cause information of
the mechanisms A and B.

The mechanism A over the purview CDE is completely reducible to (A|CD) x ([]|E) because E has no
effect on A, there is no output returned from the subsystem.mip_past function,

   >>> subsystem.mip_past(A, CDE)

Instead, we should evaluate A over the purview CD,

   >>> mip_A = subsystem.mip_past(A, CD)

In this case, there is a well defined MIP

   >>> mip_A.partition
   (Part(mechanism=(), purview=(n2,)), Part(mechanism=(n0,), purview=(n3,)))

which is ([]|C) x (A|D). It has partitioned repertoire

   >>> mip_A.partitioned_repertoire.flatten(order='F')
   array([ 0.33333333,  0.33333333,  0.16666667,  0.16666667])

and irreducible cause information

   >>> mip_A.phi
   0.16666700000000023

A similar result holds for B. Thus the mechanisms A and B exist at levels of phi=1/6, while
the higher-order mechanism AB exists only as the residual of causes, at a level of phi=1/10.

"""


def residue_subsystem():
    """ The subsystem of the residue network containing all nodes. Current and past state
    are all off (0,0,0,0,0)

    Diagram:

            +~~~~~~~~+           +~~~~~~~~+
        +~~>|   A    |<~~+  +~~~>|   B    |<~~~+
        |   | (AND)  |   |  |    | (AND)  |    |
        |   +~~~~~~~~+   |  |    +~~~~~~~~+    |
        |                |  |                  |
        |                |  |                  |
        |                |  |                  |
        |                |  |                  |
    +~~~~~~~~+        +~~~~~~~~+        +~~~~~~~~+
    |   C    |        |   D    |        |   E    |
    |        |        |        |        |        |
    +~~~~~~~~+        +~~~~~~~~+        +~~~~~~~~+

    Connectivity matrix:
    +---+---+---+---+---+---+
    | . | A | B | C | D | E |
    +---+---+---+---+---+---+
    | A | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | B | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | C | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | D | 1 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | E | 0 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+

    """

    tpm = np.array([[int(s) for s in bin(x)[2:].zfill(5)[::-1]] for x in range(32)])
    tpm[np.where(np.sum(tpm[0:,2:4],1)==2),0] = 1
    tpm[np.where(np.sum(tpm[0:,3:5],1)==2),1] = 1
    tpm[np.where(np.sum(tpm[0:,2:4],1)<2),0] = 0
    tpm[np.where(np.sum(tpm[0:,3:5],1)<2),1] = 0

    cm = np.zeros((5,5))
    cm[2:4, 0] = 1
    cm[3:, 1] = 1

    current_state = (0,0,0,0,0)
    past_state = (0,0,0,0,0)

    network = Network(tpm, current_state, past_state, connectivity_matrix=cm)
    return Subsystem(range(network.size), network)
