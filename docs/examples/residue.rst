Residue
=======

This example describes a system containing two AND gates, |A| and |B|, with a
single overlapping input node.

First let's create the subsystem corresponding to the residue network, with all
nodes OFF in the current and previous states.

    >>> import pyphi
    >>> subsystem = pyphi.examples.residue_subsystem()

Next, we can define the mechanisms of interest. Mechanisms and purviews are
represented by tuples of node indices in the network:

    >>> A = (0,)
    >>> B = (1,)
    >>> AB = (0, 1)

And the possible cause purviews that we're interested in:

    >>> CD = (2, 3)
    >>> DE = (3, 4)
    >>> CDE = (2, 3, 4)

We can then evaluate the cause information for each of the mechanisms over the
cause purview |CDE|.

    >>> subsystem.cause_info(A, CDE)
    0.333332

    >>> subsystem.cause_info(B, CDE)
    0.333332

    >>> subsystem.cause_info(AB, CDE)
    0.5

The composite mechanism |AB| has greater cause information than either of the
individual mechanisms. This contradicts the idea that |AB| should exist
minimally in this system.

Instead, we can quantify existence as the irreducible cause information of a
mechanism. The MIP of a mechanism is the partition of mechanism and purview
which makes the least difference to the cause repertoire (see the documentation
for the |RepertoireIrreducibilityAnalysis| object). The irreducible cause
information is the distance between the unpartitioned and partitioned
repertoires.

To analyze the irreducibility of the mechanism |AB| on the cause side:

    >>> mip_AB = subsystem.cause_mip(AB, CDE)

We can then determine what the specific partition is.

    >>> mip_AB.partition  # doctest: +NORMALIZE_WHITESPACE
     ∅    A,B
    ─── ✕ ───
     C    D,E

The indices ``(0, 1, 2, 3, 4)`` correspond to nodes :math:`A, B, C, D, E`
respectively. Thus the MIP is |(AB / DE) x (∅ / C)|, where :math:`\varnothing`
denotes the empty mechanism.

The partitioned repertoire of the MIP can also be retrieved:

    >>> mip_AB.partitioned_repertoire
    array([[[[[0.2, 0.2],
              [0.1, 0. ]],
    <BLANKLINE>
             [[0.2, 0.2],
              [0.1, 0. ]]]]])

And we can then calculate the irreducible cause information as the difference
between partitioned and unpartitioned repertoires.

    >>> mip_AB.phi
    0.1

One counterintuitive result that merits discussion is that since irreducible
cause information is what defines existence, we must also evaluate the
irreducible cause information of the mechanisms |A| and |B|.

The mechanism |A| over the purview |CDE| is completely reducible to
|(A / CD) x (∅ / E)| because |E| has no effect on |A|, so it has zero
|small_phi|.

    >>> subsystem.cause_mip(A, CDE).phi
    0.0
    >>> subsystem.cause_mip(A, CDE).partition  # doctest: +NORMALIZE_WHITESPACE
     ∅     A
    ─── ✕ ───
     E    C,D

Instead, we should evaluate |A| over the purview |CD|.

    >>> mip_A = subsystem.cause_mip(A, CD)

In this case, there is a well-defined MIP

    >>> mip_A.partition  # doctest: +NORMALIZE_WHITESPACE
     ∅     A
    ─── ✕ ───
     C     D

which is |(∅ / C) x (A / D)|. It has partitioned repertoire

    >>> mip_A.partitioned_repertoire
    array([[[[[0.33333333],
              [0.16666667]],
    <BLANKLINE>
             [[0.33333333],
              [0.16666667]]]]])

and irreducible cause information

    >>> mip_A.phi
    0.166667

A similar result holds for |B|. Thus the mechanisms |A| and |B| exist at levels
of |small_phi = 1/6|, while the higher-order mechanism |AB| exists only as the
residual of causes, at a level of |small_phi = 1/10|.
