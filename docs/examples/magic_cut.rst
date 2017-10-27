Magic Cuts
==========

This example explores a system of three fully connected elements |A|, |B| and
|C|, which follow the logic of the Rule 110 cellular automaton. The point of
this example is to highlight an unexpected behaviour of system cuts: that the
minimum information partition of a system can result in new concepts being
created.

First let's create the the Rule 110 network, with all nodes off in the current
state.

    >>> import pyphi
    >>> network = pyphi.examples.rule110_network()
    >>> state = (0, 0, 0)

Next, we want to identify the spatial scale and main complex of the network:

    >>> macro = pyphi.macro.emergence(network, state)
    >>> print(macro.emergence)
    -1.112671

Since the emergence value is negative, there is no macro scale which has
greater integrated information than the original micro scale. We can now
analyze the micro scale to determine the main complex of the system:

    >>> main_complex = pyphi.compute.main_complex(network, state)
    >>> main_complex.subsystem
    Subsystem(A, B, C)
    >>> print(main_complex.phi)
    1.35708

The main complex of the system contains all three nodes of the system, and it
has integrated information :math:`\Phi = 1.35708`. Now that we have identified
the main complex of the system, we can explore its conceptual structure and the
effect of the MIP.

    >>> constellation = main_complex.unpartitioned_constellation

There two equivalent cuts for this system; for concreteness we sever all
connections from elements |A| and |B| to |C|.

    >>> cut = pyphi.models.Cut(from_nodes=(0, 1), to_nodes=(2,))
    >>> cut_subsystem = pyphi.Subsystem(network, state, range(network.size),
    ...                                 cut=cut)
    >>> cut_constellation = pyphi.compute.constellation(cut_subsystem)

Let's investigate the concepts in the unpartitioned constellation,

    >>> constellation.labeled_mechanisms
    [['A'], ['B'], ['C'], ['A', 'B'], ['A', 'C'], ['B', 'C']]
    >>> constellation.phis
    [0.125, 0.125, 0.125, 0.499999, 0.499999, 0.499999]
    >>> print(sum(_))
    1.874997

and also the concepts of the partitioned constellation.

    >>> cut_constellation.labeled_mechanisms
    [['A'], ['B'], ['C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C']]
    >>> cut_constellation.phis
    [0.125, 0.125, 0.125, 0.499999, 0.266666, 0.333333]
    >>> print(sum(_))
    1.474998

The unpartitioned constellation includes all possible first and second order
concepts, but there is no third order concept. After applying the cut and
severing the connections from |A| and |B| to |C|, the third order concept |ABC|
is created and the second order concept |AC| is destroyed. The overall amount
of |small_phi| in the system decreases from :math:`1.875` to :math:`1.475`.

Let's explore the concept which was created to determine why it does not exist
in the unpartitioned constellation and what changed in the partitioned
constellation.

    >>> subsystem = main_complex.subsystem
    >>> ABC = subsystem.node_indices
    >>> subsystem.cause_info(ABC, ABC)
    0.749999
    >>> subsystem.effect_info(ABC, ABC)
    1.875

The mechanism has cause and effect power over the system, so it must be that
this power is reducible.

    >>> mice_cause = subsystem.core_cause(ABC)
    >>> mice_cause.phi
    0.0
    >>> mice_effect = subsystem.core_effect(ABC)
    >>> mice_effect.phi
    0.625

The reason ABC does not exist as a concept is that its cause is reducible.
Looking at the TPM of the system, there are no possible states with two of the
elements set to off. This means that knowing two elements are off is enough to
know that the third element must also be off, and thus the third element can
always be cut from the concept without a loss of information. This will be true
for any purview, so the cause information is reducible.

    >>> BC = (1, 2)
    >>> A = (0,)
    >>> repertoire = subsystem.cause_repertoire(ABC, ABC)
    >>> cut_repertoire = subsystem.cause_repertoire(BC, ABC) * subsystem.cause_repertoire(A, ())
    >>> pyphi.distance.hamming_emd(repertoire, cut_repertoire)
    0.0

Next, let's look at the cut subsystem to understand how the new concept comes
into existence.

    >>> ABC = (0, 1, 2)
    >>> C = (2,)
    >>> AB = (0, 1)

The cut applied to the subsystem severs the connections from |A| and |B| to
|C|. In this circumstance, knowing |A| and |B| do not tell us anything about
the state of |C|, only the past state of |C| can tell us about the future state
of |C|. Here, ``past_tpm[1]`` gives us the probability of C being on in the
next state, while ``past_tpm[0]`` would give us the probability of C being off.

    >>> C_node = cut_subsystem.indices2nodes(C)[0]
    >>> C_node.tpm_on.flatten()
    array([ 0.5 ,  0.75])

This states that A has a 50% chance of being on in the next state if it
currently off, but a 75% chance of being on in the next state  if it is
currently on. Thus unlike the unpartitioned case, knowing the current state of
C gives us additional information over and above knowing A and B.

    >>> repertoire = cut_subsystem.cause_repertoire(ABC, ABC)
    >>> cut_repertoire = (cut_subsystem.cause_repertoire(AB, ABC) *
    ...                   cut_subsystem.cause_repertoire(C, ()))
    >>> print(pyphi.distance.hamming_emd(repertoire, cut_repertoire))
    0.500001

With this partition, the integrated information is :math:`\varphi = 0.5`, but
we must check all possible partitions to find the MIP.

    >>> cut_subsystem.core_cause(ABC).purview
    (0, 1, 2)
    >>> cut_subsystem.core_cause(ABC).phi
    0.333333

It turns out that the MIP is

.. math::
   \frac{AB}{[\,]} \times \frac{C}{ABC}

and the integrated information of ABC is :math:`\varphi = 1/3`.

Note that in order for a new concept to be created by a cut, there must be a
within-mechanism connection severed by the cut.

In the previous example, the MIP created a new concept, but the amount of
|small_phi| in the constellation still decreased. This is not always the case.
Next we will look at an example of system whoes MIP increases the amount of
|small_phi|. This example is based on a five node network which follows the
logic of the Rule 154 cellular automaton. Let's first load the network,

    >>> network = pyphi.examples.rule154_network()
    >>> state = (1, 0, 0, 0, 0)

For this example, it is the subsystem consisting of |A|, |B|, and |E| that we
explore. This is not the main concept of the system, but it serves as a proof
of principle regardless.

    >>> subsystem = pyphi.Subsystem(network, state, (0, 1, 4))

Calculating the MIP of the system,

    >>> mip = pyphi.compute.big_mip(subsystem)
    >>> mip.phi
    0.217829
    >>> mip.cut
    Cut [0, 4] ━━/ /━━➤ [1]

This subsystem has a |big_phi| value of 0.15533, and the MIP cuts the
connections from |AE| to |B|. Investigating the concepts in both the
partitioned and unpartitioned constellations,

    >>> mip.unpartitioned_constellation.labeled_mechanisms
    [['A'], ['B'], ['A', 'B']]
    >>> mip.unpartitioned_constellation.phis
    [0.25, 0.166667, 0.178572]
    >>> print(sum(_))
    0.5952390000000001

The unpartitioned constellation has mechanisms |A|, |B| and |AB| with
:math:`\sum\varphi = 0.595239`.

    >>> mip.partitioned_constellation.labeled_mechanisms
    [['A'], ['B'], ['A', 'B']]
    >>> mip.partitioned_constellation.phis
    [0.25, 0.166667, 0.214286]
    >>> print(sum(_))
    0.630953

The partitioned constellation has mechanisms |A|, |B| and |AB| but with
:math:`\sum\varphi = 0.630953`. There are the same number of concepts in both
constellations, over the same mechanisms; however, the partitioned
constellation has a greater |small_phi| value for the concept |AB|, resulting
in an overall greater :math:`\sum\varphi` for the partitioned constellation.

Although situations described above are rare, they do occur, so one must be
careful when analyzing the integrated information of physical systems not to
dismiss the possibility of partitions creating new concepts or increasing the
amount of |small_phi|; otherwise, an incorrect main complex may be identified.
