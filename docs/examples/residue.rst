Residue Example
===============

* :func:`pyphi.examples.residue_network`
* :func:`pyphi.examples.residue_subsystem`

This example describes a system containing two **AND** nodes, |A| and |B|, with
a single overlapping input node.

First let's create the subsystem corresponding to the residue network, with all
nodes off in the current and past states.

   >>> import pyphi
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
