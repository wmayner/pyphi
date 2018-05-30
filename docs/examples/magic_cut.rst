Magic Cuts
==========

This example explores a system of three fully connected elements |A|, |B| and
|C|, which follow the logic of the Rule 110 cellular automaton. The point of
this example is to highlight an unexpected behaviour of system cuts: that the
minimum information partition of a system can result in new concepts being
created.

First let's create the the Rule 110 network, with all nodes OFF in the current
state.

    >>> import pyphi
    >>> network = pyphi.examples.rule110_network()
    >>> state = (0, 0, 0)

Next, we want to identify the spatial scale and major complex of the network:

    >>> macro = pyphi.macro.emergence(network, state)
    >>> print(macro.emergence)
    -1.112671

Since the emergence value is negative, there is no macro scale which has
greater integrated information than the original micro scale. We can now
analyze the micro scale to determine the major complex of the system:

    >>> major_complex = pyphi.compute.major_complex(network, state)
    >>> major_complex.subsystem
    Subsystem(A, B, C)
    >>> print(major_complex.phi)
    1.35708

The major complex of the system contains all three nodes of the system, and it
has integrated information :math:`\Phi = 1.35708`. Now that we have identified
the major complex of the system, we can explore its cause-effect structure and
the effect of the MIP.

    >>> ces = major_complex.ces

There two equivalent cuts for this system; for concreteness we sever all
connections from elements |A| and |B| to |C|.

    >>> cut = pyphi.models.Cut(from_nodes=(0, 1), to_nodes=(2,))
    >>> cut_subsystem = pyphi.Subsystem(network, state, cut=cut)
    >>> cut_ces = pyphi.compute.ces(cut_subsystem)

Let's investigate the concepts in the unpartitioned cause-effect structure,

    >>> ces.labeled_mechanisms
    (['A'], ['B'], ['C'], ['A', 'B'], ['A', 'C'], ['B', 'C'])
    >>> ces.phis
    [0.125, 0.125, 0.125, 0.499999, 0.499999, 0.499999]
    >>> sum(ces.phis)
    1.8749970000000002

and also the concepts of the partitioned cause-effect structure.

    >>> cut_ces.labeled_mechanisms
    (['A'], ['B'], ['C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C'])
    >>> cut_ces.phis
    [0.125, 0.125, 0.125, 0.499999, 0.266666, 0.333333]
    >>> sum(_)
    1.4749980000000003

The unpartitioned cause-effect structure includes all possible first and second
order concepts, but there is no third order concept. After applying the cut and
severing the connections from |A| and |B| to |C|, the third order concept |ABC|
is created and the second order concept |AC| is destroyed. The overall amount
of |small_phi| in the system decreases from :math:`1.875` to :math:`1.475`.

Let's explore the concept which was created to determine why it does not exist
in the unpartitioned cause-effect structure and what changed in the partitioned
cause-effect structure.

    >>> subsystem = major_complex.subsystem
    >>> ABC = subsystem.node_indices
    >>> subsystem.cause_info(ABC, ABC)
    0.749999
    >>> subsystem.effect_info(ABC, ABC)
    1.875

The mechanism does have cause and effect power over the system. But, since it
doesn't specify a concept, it must be that this power is reducible:

    >>> mic = subsystem.mic(ABC)
    >>> mic.phi
    0.0
    >>> mie = subsystem.mie(ABC)
    >>> mie.phi
    0.625

The reason ABC does not exist as a concept is that its cause is reducible.
Looking at the TPM of the system, there are no possible states where two
elements are OFF. This means that knowing two elements are OFF is enough to
know that the third element must also be OFF, and thus the third element can
always be cut from the concept without a loss of information. This will be true
for any purview, so the cause information is reducible.

    >>> BC = (1, 2)
    >>> A = (0,)
    >>> repertoire = subsystem.cause_repertoire(ABC, ABC)
    >>> cut_repertoire = (subsystem.cause_repertoire(BC, ABC) *
    ...                   subsystem.cause_repertoire(A, ()))
    >>> pyphi.distance.hamming_emd(repertoire, cut_repertoire)
    0.0

Next, let's look at the cut subsystem to understand how the new concept comes
into existence.

    >>> ABC = (0, 1, 2)
    >>> C = (2,)
    >>> AB = (0, 1)

The cut applied to the subsystem severs the connections going to |C| from
either |A| or |B|. In this circumstance, knowing the state of |A| or |B| does
not tell us anything about the state of |C|; only the previous state of |C| can
tell us about the next state of |C|. ``C_node.tpm_on`` gives us the probability
of |C| being ON in the next state, while ``C_node.tpm_off`` would give us the
probability of |C| being OFF.

    >>> C_node = cut_subsystem.indices2nodes(C)[0]
    >>> C_node.tpm_on.flatten()
    array([0.5 , 0.75])

This states that |C| has a 50% chance of being ON in the next state if it
currently OFF, but a 75% chance of being ON in the next state  if it is
currently ON. Thus, unlike the unpartitioned case, knowing the current state of
|C| gives us additional information over and above knowing the state of |A| or
|B|.

    >>> repertoire = cut_subsystem.cause_repertoire(ABC, ABC)
    >>> cut_repertoire = (cut_subsystem.cause_repertoire(AB, ABC) *
    ...                   cut_subsystem.cause_repertoire(C, ()))
    >>> print(pyphi.distance.hamming_emd(repertoire, cut_repertoire))
    0.500001

With this partition, the integrated information is :math:`\varphi = 0.5`, but
we must check all possible partitions to find the maximally-irreducible cause:

    >>> mic = cut_subsystem.mic(ABC)
    >>> mic.purview
    (0, 1, 2)
    >>> mic.phi
    0.333333

It turns out that the MIP of the maximally-irreducible cause is

.. math::
   \frac{AB}{\varnothing} \times \frac{C}{ABC}

and the integrated information of mechanism |ABC| is :math:`\varphi = 1/3`.

Note that in order for a new concept to be created by a cut, there must be a
within-mechanism connection severed by the cut.

In the previous example, the MIP created a new concept, but the amount of
|small_phi| in the cause-effect structure still decreased. This is not always
the case. Next we will look at an example of system whoes MIP increases the
amount of |small_phi|. This example is based on a five-node network that
implements the logic of the Rule 154 cellular automaton. Let's first load the
network:

    >>> network = pyphi.examples.rule154_network()
    >>> state = (1, 0, 0, 0, 0)

For this example, it is the subsystem consisting of |A|, |B|, and |E| that we
explore. This is not the major complex of the system, but it serves as a proof
of principle regardless.

    >>> subsystem = pyphi.Subsystem(network, state, (0, 1, 4))

Calculating the MIP of the system,

    >>> sia = pyphi.compute.sia(subsystem)
    >>> sia.phi
    0.217829
    >>> sia.cut
    Cut [A, E] ━━/ /━━➤ [B]

we see that this subsystem has a |big_phi| value of 0.15533, and the MIP cuts
the connections from |AE| to |B|. Investigating the concepts in both the
partitioned and unpartitioned cause-effect structures,

    >>> sia.ces.labeled_mechanisms
    (['A'], ['B'], ['A', 'B'])
    >>> sia.ces.phis
    [0.25, 0.166667, 0.178572]
    >>> print(sum(_))
    0.5952390000000001

We see that the unpartitioned cause-effect structure has mechanisms |A|, |B|
and |AB| with :math:`\sum\varphi = 0.595239`.

    >>> sia.partitioned_ces.labeled_mechanisms
    (['A'], ['B'], ['A', 'B'])
    >>> sia.partitioned_ces.phis
    [0.25, 0.166667, 0.214286]
    >>> print(sum(_))
    0.630953

The partitioned cause-effect structure has mechanisms |A|, |B| and |AB| but
with :math:`\sum\varphi = 0.630953`. There are the same number of concepts in
both cause-effect structures, over the same mechanisms; however, the
partitioned cause-effect structure has a greater |small_phi| value for the
concept |AB|, resulting in an overall greater :math:`\sum\varphi` for the
partitioned cause-effect structure.

Although situations described above are rare, they do occur, so one must be
careful when analyzing the integrated information of physical systems not to
dismiss the possibility of partitions creating new concepts or increasing the
amount of |small_phi|; otherwise, an incorrect major complex may be identified.
