XOR Network
===========

* :func:`pyphi.examples.xor_network`
* :func:`pyphi.examples.xor_subsystem`

This example describes a system of three fully connected **XOR** nodes, |n0|,
|n1| and |n2| (no self-connections).

First let's create the XOR network, with all nodes **OFF** in both the current
and past states.

    >>> import pyphi
    >>> network = pyphi.examples.xor_network()

Existence is a top-down process; the whole is more important than its parts.
The first step is to confirm the existence of the whole, by finding the main
complex of the network:

    >>> main_complex = pyphi.compute.main_complex(network)

The main complex exists (|Phi > 0|),

    >>> main_complex.phi
    1.874999

and it consists of the entire network:

    >>> main_complex.subsystem
    Subsystem((n0, n1, n2))

Knowing what exists at the system level, we can now investigate the existence
of concepts within the complex.

    >>> constellation = main_complex.unpartitioned_constellation
    >>> len(constellation)
    3
    >>> [concept.mechanism for concept in constellation]
    [(n0, n1), (n0, n2), (n1, n2)]

There are three concepts in the constellation. They are all the possible
second order mechanisms: |n0, n1|, |n0, n2| and |n1, n2|.

Focusing on the concept specified by mechanism |n0, n1|, we investigate
existence, and the irreducible cause and effect. Based on the symmetry of the
network, the results will be similar for the other second order mechanisms.

    >>> concept = constellation[0]
    >>> concept.mechanism
    (n0, n1)
    >>> concept.phi
    0.5

The concept has :math:`\varphi = \frac{1}{2}`.

    >>> concept.cause.purview
    (n0, n1, n2)
    >>> concept.cause.repertoire
    array([[[ 0.5,  0. ],
            [ 0. ,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0. ],
            [ 0. ,  0.5]]])

So we see that the cause purview of this mechanism is the whole system |n0, n1,
n2|, and that the repertoire shows a :math:`0.5` of probability the past state
being ``(0, 0, 0)`` and the same for ``(1, 1, 1)``:

    >>> concept.cause.repertoire[(0, 0, 0)]
    0.5
    >>> concept.cause.repertoire[(1, 1, 1)]
    0.5

This tells us that knowing both |n0| and |n1| are currently **OFF** means that
the past state of the system was either all **OFF** or all **ON** with equal
probability.

For any reduced purview, we would still have the same information about the
elements in the purview (either all **ON** or all **OFF**), but we would lose
the information about the elements outside the purview.

    >>> concept.effect.purview
    (n2,)
    >>> concept.effect.repertoire
    array([[[ 1.,  0.]]])

The effect purview of this concept is the node |n2|. The mechanism |n0, n1| is
able to completely specify the next state of |n2|. Since both nodes are
**OFF**, the next state of |n2| will be **OFF**.

The mechanism |n0, n1| does not provide any information about the next state of
either |n0| or |n1|, because the relationship depends on the value of |n2|.
That is, the next state of |n0| (or |n1|) may be either **ON** or **OFF**,
depending on the value of |n2|. Any purview larger than |n2| would be reducible
by pruning away the additional elements.

+--------------------------------------------------------------------------+
| Main Complex: |n0, n1, n2| with :math:`\Phi = 1.875`                     |
+===============+=================+===================+====================+
| **Mechanism** | :math:`\varphi` | **Cause Purview** | **Effect Purview** |
+---------------+-----------------+-------------------+--------------------+
| |n0, n1|      |  0.5            | |n0, n1, n2|      | |n2|               |
+---------------+-----------------+-------------------+--------------------+
| |n0, n2|      |  0.5            | |n0, n1, n2|      | |n1|               |
+---------------+-----------------+-------------------+--------------------+
| |n1, n2|      |  0.5            | |n0, n1, n2|      | |n0|               |
+---------------+-----------------+-------------------+--------------------+

An analysis of the `intrinsic existence` of this system reveals that the main
complex of the system is the entire network of **XOR** nodes. Furthermore, the
concepts which exist within the complex are those specified by the second-order
mechanisms |n0, n1|, |n0, n2|, and |n1, n2|.

To understand the notion of intrinsic existence, in addition to determining
what exists for the system, it is useful to consider also what does not exist.

Specifically, it may be surprising that none of the first order mechanisms
|n0|, |n1| or |n2| exist. This physical system of **XOR** gates is sitting on
the table in front of me; I can touch the individual elements of the system, so
how can it be that they do not exist?

That sort of existence is what we term `extrinsic existence`. The **XOR** gates
exist for me as an observer, external to the system. I am able to manipulate
them, and observe their causes and effects, but the question that matters for
`intrinsic` existence is, do they have irreducible causes and effects within
the system? There are two reasons a mechanism may have no irreducible
cause-effect power: either the cause-effect power is completely reducible, or
there was no cause-effect power to begin with. In the case of elementary
mechanisms, it must be the latter.

To see this, again due to symmetry of the system, we will focus only on the
mechanism |n0|.

   >>> subsystem = pyphi.examples.xor_subsystem()
   >>> n0 = (subsystem.nodes[0],)
   >>> n0n1n2 = subsystem.nodes

In order to exist, a mechanism must have irreducible cause and effect power
within the system.

   >>> subsystem.cause_info(n0, n0n1n2)
   0.5
   >>> subsystem.effect_info(n0, n0n1n2)
   0.0

The mechanism has no effect power over the entire subsystem, so it cannot have
effect power over any purview within the subsystem. Furthermore, if a mechanism
has no effect power, it certainly has no irreducible effect power. The
first-order mechanisms of this system do not exist intrinsically, because they
have no effect power (having causal power is not enough).

To see why this is true, consider the effect of |n0|. There is no self-loop, so
|n0| can have no effect on itself. Without knowing the current state of |n0|,
in the next state |n1| could be either **ON** or **OFF**. If we know that the
current state of |n0| is **ON**, then |n1| could still be either **ON** or
**OFF**, depending on the state of |n2|. Thus, on its own, the current state of
|n0| does not provide any information about the next state of |n1|. A similar
result holds for the effect of |n0| on |n2|. Since |n0| has no effect power
over any element of the system, it does not exist from the intrinsic
perspective.

To complete the discussion, we can also investigate the potential third order
mechanism |n0, n1, n2|. Consider the cause information over the purview |n0,
n1, n2|:

   >>> subsystem.cause_info(n0n1n2, n0n1n2)
   0.749999

Since the mechanism has nonzero cause information, it has causal power over the
system—but is it irreducible?

   >>> mip = subsystem.mip_past(n0n1n2, n0n1n2)
   >>> mip.phi
   0.0
   >>> mip.partition
   (Part(mechanism=(n0,), purview=()), Part(mechanism=(n1, n2), purview=(n0, n1, n2)))

The mechanism has :math:`ci = 0.75`, but it is completely reducible
(:math:`\varphi = 0`) to the partition 

.. math::
    \frac{n0}{[\left\,\right]} \times \frac{n_1n_2}{n_0n_1n_2}

This result can be understood as follows: knowing that |n1| and |n2| are
**OFF** in the current state is sufficient to know that |n0|, |n1|, and |n2|
were all **OFF** in the past state; there is no additional information gained
by knowing that |n0| is currently **OFF**.

Similarly for any other potential purview, the current state of |n1| and |n2|
being ``(0, 0)`` is always enough to fully specify the previous state, so the
mechanism is reducible for all possible purviews, and hence does not exist.
