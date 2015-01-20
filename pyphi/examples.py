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


**Basic Example**


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
structures that are calculated in the course of arriving at a final |big_phi|
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
    :func:`pyphi.examples.basic_network` and
    :func:`pyphi.examples.basic_subsystem` functions.

-------------------------------------------------------------------------------


**Residue Example**


This example describes a system containing two **AND** nodes, |A| and |B|, with
a single overlapping input node.

First let's create the subsystem corresponding to the residue network, with all
nodes off in the current and past states.

   >>> subsystem = pyphi.examples.residue_subsystem()

Next, we can define the mechanisms of interest:

   >>> A = subsystem.indices2nodes((0,))
   >>> B = subsystem.indices2nodes((1,))
   >>> AB = subsystem.indices2nodes((0, 1))

And the possible past purviews that we're interested in:

   >>> CD = subsystem.indices2nodes((2, 3))
   >>> DE = subsystem.indices2nodes((3, 4))
   >>> CDE = subsystem.indices2nodes((2, 3, 4))

We can then evaluate the cause information for each of the mechanisms over the
past purview |CDE|.

   >>> subsystem.cause_info(A, CDE)
   0.33333191666400036

   >>> subsystem.cause_info(B, CDE)
   0.33333191666400036

   >>> subsystem.cause_info(AB, CDE)
   0.49999972500000006

The composite mechanism |AB| has greater cause information than either of the
individual mechanisms. This contradicts the idea that |AB| should exist
minimally in this system.

Instead, we can quantify existence as the irreducible cause information of a
mechanism. The **MIP** of a mechanism is the partition of mechanism and purview
which makes the least difference to the cause repertoire (see
:class:`pyphi.models.Mip`). The irreducible cause information is the distance
between the unpartitioned and partitioned repertoires.

To calculate the MIP structure of mechanism |AB|:

   >>> mip_AB = subsystem.mip_past(AB, CDE)

We can then determine what the specific partition is.

   >>> mip_AB.partition
   (Part(mechanism=(), purview=(n2,)), Part(mechanism=(n0, n1), purview=(n3, n4)))

The labels ``(n0, n1, n2, n3, n4)`` correspond to nodes :math:`A, B, C, D, E`
respectively. Thus the MIP is |(AB / DE) x ([] / C)|, where :math:`[]` denotes
the empty mechanism.

The partitioned repertoire of the MIP can also be retrieved:

   >>> mip_AB.partitioned_repertoire.flatten(order='F')
   array([ 0.2,  0.2,  0.1,  0.1,  0.2,  0.2,  0. ,  0. ])

And we can then calculate the irreducible cause information as the difference
between partitioned and unpartitioned repertoires.

   >>> mip_AB.phi
   0.09999990000000035

One counterintuitive result which merits discussion is that since irreducible
cause information is what defines existence, we must also evaluate the
irreducible cause information of the mechanisms |A| and |B|.

The mechanism |A| over the purview |CDE| is completely reducible to |(A / CD) x
([] / E)| because |E| has no effect on |A|, so it has zero |small_phi|.

   >>> subsystem.mip_past(A, CDE).phi
   0.0
   >>> subsystem.mip_past(A, CDE).partition
   (Part(mechanism=(), purview=(n4,)), Part(mechanism=(n0,), purview=(n2, n3)))

Instead, we should evaluate |A| over the purview |CD|.

   >>> mip_A = subsystem.mip_past(A, CD)

In this case, there is a well defined MIP

   >>> mip_A.partition
   (Part(mechanism=(), purview=(n2,)), Part(mechanism=(n0,), purview=(n3,)))

which is |([] / C) x (A / D)|. It has partitioned repertoire

   >>> mip_A.partitioned_repertoire.flatten(order='F')
   array([ 0.33333333,  0.33333333,  0.16666667,  0.16666667])

and irreducible cause information

   >>> mip_A.phi
   0.16666700000000023

A similar result holds for |B|. Thus the mechanisms |A| and |B| exist at levels
of |small_phi = 1/6|, while the higher-order mechanism |AB| exists only as the
residual of causes, at a level of |small_phi = 1/10|.

-------------------------------------------------------------------------------


**Conditional Independence Example**

This example explores the assumption of conditional independence, and the
behaviour of the program when it is not satisfied.

Every state-by-node tpm corresponds to a unique state-by-state tpm
which satisfies the conditional independence assumption. If a
state-by-node tpm is given as input for a network, the program assumes
that it is from a system with the corresponding conditionally independent
state-by-state tpm.

When a state-by-state tpm is given as input for a network, the state-by-state
tpm is first converted to a state-by-node tpm. The program then assumes that
the system corresponds to the unique conditionally independent representation of
the state-by-node tpm. If a non-conditionally independent tpm is given, the analyzed
system will not correspond to the original tpm. (Note that every deterministic
state-by-state tpm will automatically satisfy the conditional independence
assumption.)

Consider a system of two binary nodes(A and B) which do not change if
they have the same value, but flip with probability 50% if they have
different values.

Load the state_by_state tpm for such a system,

   >>> tpm = pyphi.examples.cond_depend_tpm()
   >>> print(tpm)
   [[ 1.   0.   0.   0. ]
    [ 0.   0.5  0.5  0. ]
    [ 0.   0.5  0.5  0. ]
    [ 0.   0.   0.   1. ]]

This system does not satisfy the conditional independence assumption;
given a past state of (1,0) the current state of node A depends on
whether or not B has flipped.

When creating a network, the program will convert this state-by-state
tpm to a state-by-node form, and issue a warning if it does not
satisfy the assumption,

   >>> sbn_tpm = pyphi.convert.state_by_state2state_by_node(tpm)


 Warning: The tpm is not conditionally independent, see documentation of pyphi.Examples
 for more information on how this is handled

   >>> print(sbn_tpm)
   [[[ 0.   0. ]
     [ 0.5  0.5]]
   <BLANKLINE>
    [[ 0.5  0.5]
     [ 1.   1. ]]]

The program will continue with the state-by-node tpm, but since it assumes
conditional independence, it does not correspond to the original system.

To see the corresponding conditionally independent tpm, convert the state-by-node
tpm back to state-by-state form,

   >>> sbs_tpm = pyphi.convert.state_by_node2state_by_state(sbn_tpm)
   >>> print(sbs_tpm)
   [[ 1.    0.    0.    0.  ]
    [ 0.25  0.25  0.25  0.25]
    [ 0.25  0.25  0.25  0.25]
    [ 0.    0.    0.    1.  ]]


A system which does not satisfy the conditional independence assumption shows 'instantaneous causality'
In such situations, there must be additional exogenous variable(s) which explain the dependence.

Consider the above example, except that there is a third node (C) which is equally likely to be ON or OFF.
When nodes A and B are in different states, they will flip when C is ON, but stay the same when C is OFF.

   >>> tpm2 = pyphi.examples.cond_independ_tpm()
   >>> print(tpm2)
   [[ 0.5  0.   0.   0.   0.5  0.   0.   0. ]
    [ 0.   0.5  0.   0.   0.   0.5  0.   0. ]
    [ 0.   0.   0.5  0.   0.   0.   0.5  0. ]
    [ 0.   0.   0.   0.5  0.   0.   0.   0.5]
    [ 0.5  0.   0.   0.   0.5  0.   0.   0. ]
    [ 0.   0.   0.5  0.   0.   0.   0.5  0. ]
    [ 0.   0.5  0.   0.   0.   0.5  0.   0. ]
    [ 0.   0.   0.   0.5  0.   0.   0.   0.5]]

The resulting state-by-state tpm now satisfies the conditional independence
assumption.

   >>> sbn_tpm2 = pyphi.convert.state_by_state2state_by_node(tpm2)
   >>> print(sbn_tpm2)
   [[[[ 0.   0.   0.5]
      [ 0.   0.   0.5]]
   <BLANKLINE>
     [[ 0.   1.   0.5]
      [ 1.   0.   0.5]]]
   <BLANKLINE>
   <BLANKLINE>
    [[[ 1.   0.   0.5]
      [ 0.   1.   0.5]]
   <BLANKLINE>
     [[ 1.   1.   0.5]
      [ 1.   1.   0.5]]]]

The node indices are [0,1] for AB and [2] for C,

   >>> AB = np.array([0,1])
   >>> C = np.array([2])

From here, if we marginalize out the node C

   >>> tpm2_marginalizeC = pyphi.utils.marginalize_out(C, sbn_tpm2)

And then restrict the purview to only nodes A and B

   >>> tpm2_purviewAB = np.squeeze(tpm2_marginalizeC[:,:,:,AB])

We get back the original state-by-node tpm from the system with
just A and B

   >>> np.all(tpm2_purviewAB == sbn_tpm)
   True

-------------------------------------------------------------------------------


A note on conventions
---------------------

**TPMs**


There are several ways to write down a TPM. With both state-by-state and
state-by-node TPMs, one is confronted with a choice about which rows correspond
to which states. In state-by-state TPMs, this choice must also be made for the
columns.

Either the first node changes state every other row:

    +------+-----+-----+
    | A, B |  A  |  B  |
    +======+=====+=====+
    | 0, 0 | 0.1 | 0.2 |
    +------+-----+-----+
    | 1, 0 | 0.3 | 0.4 |
    +------+-----+-----+
    | 0, 1 | 0.5 | 0.6 |
    +------+-----+-----+
    | 1, 1 | 0.7 | 0.8 |
    +------+-----+-----+

Or the last node does:

    +------+-----+-----+
    | A, B |  A  |  B  |
    +======+=====+=====+
    | 0, 0 | 0.1 | 0.2 |
    +------+-----+-----+
    | 0, 1 | 0.5 | 0.6 |
    +------+-----+-----+
    | 1, 0 | 0.3 | 0.4 |
    +------+-----+-----+
    | 1, 1 | 0.7 | 0.8 |
    +------+-----+-----+

Note that the index |i| of a row in a TPM encodes a network state: convert the
index to binary, and each bit gives the state of a node. The question is, which
node?

**Throughout PyPhi, we always choose the first convention—the first node's
state varies the fastest.**

Since numbers are written with the least-significant digit on the right, the
right-most digit varies the fastest. This means that according to our
convention, **the right-most (least-significant) bit gives the state of the
first node**.

We call this convention the **LOLI convention**: Low Order bits correspond to
Low Index nodes. The other convention, where the highest-index node varies the
fastest, is similarly called **HOLI**.

.. note::

    The rationale for this choice of convention is that the **LOLI** mapping is
    stable under changes in the number of nodes, in the sense that the same bit
    always corresponds to the same node index. The **HOLI** mapping does not
    have this property.

.. note::

    This obviously applies to only situations where decimal indices are
    encoding states. Whenever a network state is represented as a list or
    tuple, we use the only sensible convention: the |ith| element gives the
    state of the |ith| node.

.. note::

    There are various conversion functions available for converting between
    TPMs, states, and indices using different conventions: see the
    :mod:`pyphi.convert` module.

-------------------------------------------------------------------------------


**Connectivity Matrices**


Throughout PyPhi, if ``CM`` is a connectivity matrix, then |CM[i][j] = 1| means
that node |i| is connected to node |j|.

-------------------------------------------------------------------------------
"""

import numpy as np
from .network import Network
from .subsystem import Subsystem


def basic_network():
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
    |  Past state  | Current state |
    +--------------+---------------+
    |   A, B, C    |    A, B, C    |
    +==============+===============+
    |   0, 0, 0    |    0, 0, 0    |
    +--------------+---------------+
    |   1, 0, 0    |    0, 0, 1    |
    +--------------+---------------+
    |   0, 1, 0    |    1, 0, 1    |
    +--------------+---------------+
    |   1, 1, 0    |    1, 0, 0    |
    +--------------+---------------+
    |   0, 0, 1    |    1, 1, 0    |
    +--------------+---------------+
    |   1, 0, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   0, 1, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   1, 1, 1    |    1, 1, 0    |
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
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ])

    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (1, 0, 0)
    past_state = (1, 1, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def basic_subsystem():
    """A subsystem containing all the nodes of the
    :func:`pyphi.examples.basic_network`."""
    net = basic_network()
    return Subsystem(range(net.size), net)


def residue_network():
    """The network for the residue example.

    Current and past state are all nodes off.

    Diagram::

                +~~~~~~~+         +~~~~~~~+
                |   A   |         |   B   |
            +~~>| (AND) |         | (AND) |<~~+
            |   +~~~~~~~+         +~~~~~~~+   |
            |        ^               ^        |
            |        |               |        |
            |        +~~~~~+   +~~~~~+        |
            |              |   |              |
        +~~~+~~~+        +~+~~~+~+        +~~~+~~~+
        |   C   |        |   D   |        |   E   |
        |       |        |       |        |       |
        +~~~~~~~+        +~~~~~~~+        +~~~~~~~+

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
    tpm = np.array([
        [int(s) for s in bin(x)[2:].zfill(5)[::-1]] for x in range(32)
    ])
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) == 2), 0] = 1
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) == 2), 1] = 1
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) < 2), 0] = 0
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) < 2), 1] = 0

    cm = np.zeros((5, 5))
    cm[2:4, 0] = 1
    cm[3:, 1] = 1

    current_state = (0, 0, 0, 0, 0)
    past_state = (0, 0, 0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def residue_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.residue_network`."""
    net = residue_network()
    return Subsystem(range(net.size), net)


def xor_network():
    """A fully connected system of three XOR gates. In the state ``(0, 0, 0)``,
    none of the elementary mechanisms exist.

    Diagram::

        +~~~~~~~+       +~~~~~~~+
        |   A   +<~~~~~>|   B   |
        | (XOR) |       | (XOR) |
        +~~~~~~~+       +~~~~~~~+
            ^               ^
            |   +~~~~~~~+   |
            +~~>|   C   |<~~+
                | (XOR) |
                +~~~~~~~+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])

    cm = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (0, 0, 0)
    past_state = (0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def xor_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.xor_network`."""
    net = xor_network()
    return Subsystem(range(net.size), net)


def cond_depend_tpm():
    """A system of two general logic gates A and B such if they are in the same
    state they stay the same, but if they are in different states, they flip
    with probability 50%.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+

    TPM:

    +------+------+------+------+------+
    |      |(0, 0)|(1, 0)|(0, 1)|(1, 1)|
    +------+------+------+------+------+
    |(0, 0)| 1.0  | 0.0  | 0.0  | 0.0  |
    +------+------+------+------+------+
    |(1, 0)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(0, 1)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(1, 1)| 0.0  | 0.0  | 0.0  | 1.0  |
    +------+------+------+------+------+

    Connectivity matrix:

    +---+---+---+
    | . | A | B |
    +---+---+---+
    | A | 0 | 1 |
    +---+---+---+
    | B | 1 | 0 |
    +---+---+---+
    """

    tpm = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return tpm


def cond_independ_tpm():
    """A system of three general logic gates A, B and C such that if A and B
    are in the same state then they stay the same. If they are in different
    states, they flip if C is ''ON and stay the same if C is OFF. Node C is ON
    50% of the time, independent of the previous state.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+
           ^               ^
           |    +~~~~~+    |
           +~~~~+  C  +~~~~+
                +~~~~~+

    TPM:

    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |         |(0, 0, 0)|(1, 0, 0)|(0, 1, 0)|(1, 1, 0)|(0, 0, 1)|(1, 0, 1)|(0, 1, 1)|(1, 1, 1)|
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 0)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 0)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 0)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 0)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 1)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 1)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 1)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 1)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 0 |
    +---+---+---+---+
    | B | 1 | 0 | 0 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """

    tpm = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
    ])

    return tpm

