XOR Example
===========

* :func:`pyphi.examples.xor_network`
* :func:`pyphi.examples.xor_subsystem`

This example describes a system of three fully connected **XOR** nodes, |n0|, |n1|
and |n2| (no self connections).

First let's create the xor network, with all nodes OFF in both the current and past states.

   >>> import pyphi
   >>> network = pyphi.examples.xor_network()

Existence is a top-down process, the whole is more important than its parts.
The first step is to confirm the existence of the whole, by finding the
main complex of the network:

   >>> main_complex = pyphi.compute.main_complex(network)

The main complex exists (|Phi > 0|),

   >>> main_complex.phi
   1.8749970000011253

and is the subsystem that is the entire network,

   >>> main_complex.subsystem
   Subsystem((n0, n1, n2))

Knowing what exists at the system level, we can now investigate
the existence of concepts within the complex.

   >>> constellation = main_complex.unpartitioned_constellation
   >>> len(constellation)
   3
   >>> [concept.mechanism for concept in constellation]
   [(n0, n1), (n0, n2), (n1, n2)]

There are three concepts in the constellation, they are all the possible
second order mechanisms, |(n0, n1)|, |(n0, n2)| and |(n1, n2)|.

Focusing on the concept with mechanism (n0, n1), we investigate existence,
and the irreducible cause and effect. Based on the symmetry of the network,
the results will be similar for the other second order mechanisms.

   >>> concept = constellation[0]
   >>> concept.phi
   0.49999950000000004

The mechanism has |small_phi = 1/2|.

   >>> concept.cause.purview
   (n0, n1, n2)
   >>> concept.cause.repertoire.flatten(order='F')
   array([ 0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5])

The cause purview of this mechanism is the whole system |(n0, n1, n2)|.
Knowing that both |n0| and |n1| are currently off (0, 0) tells us that the
past state of the system was either all OFF (0, 0, 0) or  all ON (1, 1, 1).

For any reduced purview, we would still have the same information about the
elements in the purview (either all ON or all OFF), but we would lose the
information about the elements outside the purview.

   >>> concept.effect.purview
   (n2,)
   >>> concept.effect.repertoire.flatten(order='F')
   array([ 1.,  0.])

The effect purview of this concept is the node |(n2,)|. The mechanism
|(n0, n1)| is able to completely specify the next state of |n2|. Since
both nodes are OFF, the future state of |n2| will be off.

There mechanism |(n0, n1)| does not provide any information about the future
state of either |n0| or |n1|, because the relationship depends on the value
of |n2|. That is, the future state of |n0| (or |n1|) may be either ON or OFF,
depending on the value of |n2|. Any purview larger than |(n2,)| would be
reducible by pruning away the additional elements.

+----------------------------------------------------------+
| Main Complex: (n0, n1, n2)  Phi = 1.875                  |
+============+============+===============+================+
| Mechanism  | small phi  | Cause Purview | Effect Purview |
+============+============+===============+================+
| (n0, n1)   |  0.5       | (n0, n1, n2)  | (n2,)          |
+------------+------------+---------------+----------------+
| (n0, n2)   |  0.5       | (n0, n1, n2)  | (n1,)          |
+------------+------------+---------------+----------------+
| (n1, n2)   |  0.5       | (n0, n1, n2)  | (n0,)          |
+------------+------------+---------------+----------------+

An analysis of the intrinsic existence of this system reveals that
the main complex of the system is the entire network of XOR nodes.
Furthermore, the concepts which exist within the complex are the
second order mechanisms |((n0, n1), (n0, n2), (n1, n2))|.


In understanding intrinsic existence, in addition to determining what exists
for the system, it is beneficial to consider what does not exist.

Specifically, it may be surprising that none of the first order mechanisms |n0|, |n1|
or |n2| exist. This physical system of XOR gates is sitting on the table
in front of me, I can touch the individual elements of the system, so how can
it be that they do not exist?

The existence just described is an extrinsic existence. The XOR gates exist for me
as an external observed to the system. I am able to manipulate them, and observe
their causes and effects, but the question for intrinsic existence is, do that have
irreducible causes and effects within the system? There are two reasons a mechanism
may have no irreducible cause-effect power: either the cause-effect power is
completely reducible, or there was no cause-effect power to begin with. In the case
of elementary mechanisms, it must be the later.

Again due to symmetry of the system, we will focus on the mechanism |n0|.

   >>> subsystem = pyphi.examples.xor_subsystem()
   >>> n0 = (subsystem.nodes[0],)
   >>> n0n1n2 = subsystem.nodes

In order to exist, a mechanism must have irreducible cause and effect
power within the system.

   >>> subsystem.cause_info(n0, n0n1n2)
   0.49999950000000004
   >>> subsystem.effect_info(n0, n0n1n2)
   0.0

The mechanism has no effect power over the entire subsystem, so it also
has no effect power over any purview within the subsystem. Furthermore,
if a mechanism has no effect power, it certainly has no irreducible effect
power. The first order mechanisms of this system do not exist, because
they have no effect power (having cause power is not enough).

To see why this is true, consider the effect of |n0|. There is no self-loop,
so |n0| can have no effect on itself. Without knowing the current state of
|n0|, in the next state |n1| could be either ON or OFF. If we know that
the current state of |n0| is ON, then |n1| could still be either ON or OFF,
depending on the state of |n2|. Thus, on its own,  the current state of |n0|
does not provide any information about the future state of |n1|. A similar
result holds for the effect of |n0| on |n2|. Since |n0| has no effect power
over any element of the system, it does not exist from the intrinsic perspective.

To complete the discussion, we can also investigate the potential third order
mechanism |n0, n1, n2|. Consider the cause power over the purview (n0, n1, n2):

   >>> subsystem.cause_info(n0n1n2, n0n1n2)
   0.74999925

The mechanism has cause power over the system, but is it irreducible?

   >>> mip = subsystem.mip_past(n0n1n2, n0n1n2)
   >>> mip.phi
   0.0
   >>> mip.partition
   (Part(mechanism=(n0,), purview=()), Part(mechanism=(n1, n2), purview=(n0, n1, n2)))

The mechanism has cause power of |0.75|, but it is completely reducible
(|phi=0.0|) to the partition |(n0/[]) x (n1n2/n0n1n2)|. This result can
be understood as follows: knowing that (n1, n2) = (0, 0) in the current
state is sufficient to know the past state of (n0, n1, n2) = (0, 0, 0),
there is no additional information gained by knowing the current
value (n0) = (0).

Similarly for any other potential purview, the current value (n1, n2) = (0, 0)
is always enough to fully specify the previous state, so the mechanism
is reducible for all possible purviews, and hence does not exist.

