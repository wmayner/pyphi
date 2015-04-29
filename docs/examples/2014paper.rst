IIT 3.0 Paper (2014)
====================

This section is meant to serve as a companion to the paper `From the
Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory
3.0
<http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588>`_
by Oizumi, Albantakis, and Tononi, and as a demonstration of how to use PyPhi.
Readers are encouraged to follow along and analyze the systems shown in the
figures, hopefully becoming more familiar with both the theory and the software
in the process.

First, start a Python 3 `REPL
<http://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`_ by
running ``python3`` on the command line. Then import PyPhi and NumPy:

    >>> import pyphi
    >>> import numpy as np

Figure 1
~~~~~~~~

**Existence: Mechanisms in a state having causal power.**

For the first figure, we'll demonstrate how to set up a network and a candidate
set. In PyPhi, networks are built by specifying a transition probability
matrix, a past state, a current state, and (optionally) a connectivity matrix.
(If no connectivity matrix is given, full connectivity is assumed.) So, to set
up the system shown in Figure 1, we'll start by defining its TPM. 

.. note::
    The TPM in the figure is given in **state-by-state** form; there is a row
    and a column for each state. However, in PyPhi, we use a more compact
    representation: **state-by-node** form, in which there is a row for each
    state, but a column for each node. The |i,jth| entry gives the probability
    that the |jth| node is on in the |ith| state. For more information on how
    TPMs are represented in PyPhi, see the documentation for the
    :ref:`pyphi-network` module and the explanation of :ref:`loli-convention`.

In the figure, the TPM is shown only for the candidate set. We'll define the
entire network's TPM. Also, nodes :math:`D, E` and :math:`F` are not assigned
mechanisms; for the purposes of this example we will assume they are **OR**
gates. With that assumption, we get the following TPM (before copying and
pasting, see note below):

    >>> tpm = np.array([
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 0, 0, 1, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [1, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 1, 0]
    ... ])

.. note::
    This network is already built for you; you can get it from the
    :mod:`pyphi.examples` module with ``network = pyphi.examples.fig1a()``. The
    TPM can then be accessed with ``network.tpm``.

Now we'll define the current and past state:

    >>> current_state = (1, 0, 0, 0, 1, 0)
    >>> past_state = (1, 1, 0, 0, 0, 0)

Next we'll define the connectivity matrix. In PyPhi, the |i,jth| entry in a
connectivity matrix indicates whether node |i| is connected to node |j|. Thus,
this network's connectivity matrix is

    >>> cm = np.array([
    ...     [0, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 1, 0, 0, 0, 0],
    ...     [1, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0]
    ... ])

Now we can pass the TPM, current and past states, and connectivity matrix as
arguments to the network constructor (note that the current state is the second
argument and the past state is the third argument):

    >>> network = pyphi.Network(tpm, current_state, past_state,
    ...                         connectivity_matrix=cm)

Now the network shown in the figure is stored in a variable called ``network``.
You can find more information about the network object we just created by
running ``help(network)`` or by consulting the `API
<http://en.wikipedia.org/wiki/Application_programming_interface>`_
documentation here: :mod:`pyphi.network`.

The next step is to define the candidate set shown in the figure, consisting of
nodes :math:`A, B` and :math:`C`. In PyPhi, a candidate set for |big_phi|
evaluation is represented by the :class:`pyphi.Subsystem` class. Subsystems are
built by giving the indices of the nodes in the subsystem and the network it is
a part of. So, we define our candidate set like so:

    >>> ABC = pyphi.Subsystem([0, 1, 2], network)

For more information on the subsystem object, see :mod:`pyphi.subsystem`.

That covers the basic workflow with PyPhi and introduces the two types of
objects we use to represent and analyze networks. First you define the network
of interest with a TPM, current/past state, and connectivity matrix, then
you define a candidate set you want to analyze.


Figure 3
~~~~~~~~

**Information requires selectivity.**

(A)
```

We'll start by setting up the subsytem depicted in the figure and labeling the
nodes. In this case, the subsystem is just the entire network.

    >>> network = pyphi.examples.fig3a()
    >>> network.current_state
    (1, 0, 0, 0)
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C, D = subsystem.nodes

Since the connections are noisy, we see that :math:`A = 1` is unselective; all
past states are equally likely:

    >>> subsystem.cause_repertoire((A,), (B, C, D))
    array([[[[ 0.125,  0.125],
             [ 0.125,  0.125]],
    <BLANKLINE>
            [[ 0.125,  0.125],
             [ 0.125,  0.125]]]])

And this gives us zero cause information:

    >>> subsystem.cause_info((A,), (B, C, D))
    0.0


(B)
```

The same as (A) but without noisy connections:

    >>> network = pyphi.examples.fig3b()
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C, D = subsystem.nodes

Now, :math:`A`'s cause repertoire is maximally selective.

    >>> cr = subsystem.cause_repertoire((A,), (B, C, D))
    >>> cr
    array([[[[ 0.,  0.],
             [ 0.,  0.]],
    <BLANKLINE>
            [[ 0.,  0.],
             [ 0.,  1.]]]])


Since the cause repertoire is over the purview :math:`BCD`, the first dimension
(which corresponds to :math:`A`'s states) is a singleton. We can squeeze out
:math:`A`'s singleton dimension with

    >>> cr = cr.squeeze()

and now we can see that the probability of :math:`B, C,` and :math:`D` having
been all on is 1:

    >>> cr[(1, 1, 1)]
    1.0

Now the cause information specified by :math:`A = 1` is 1.5:

    >>> subsystem.cause_info((A,), (B, C, D))
    1.5


(C)
```

The same as (B) but with :math:`A = 0`:

    >>> network = pyphi.examples.fig3c()
    >>> network.current_state
    (0, 0, 0, 0)
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C, D = subsystem.nodes

And here the cause repertoire is minimally selective, only ruling out the state
where :math:`B, C,` and :math:`D` were all on:

    >>> subsystem.cause_repertoire((A,), (B, C, D))
    array([[[[ 0.14285714,  0.14285714],
             [ 0.14285714,  0.14285714]],
    <BLANKLINE>
            [[ 0.14285714,  0.14285714],
             [ 0.14285714,  0.        ]]]])

And so we have less cause information:

    >>> subsystem.cause_info((A,), (B, C, D))
    0.21428400000000067

Figure 4
~~~~~~~~

**Information: “Differences that make a difference to a system from its own
intrinsic perspective.”**

First we'll get the network from the examples module, set up a subsystem, and
label the nodes, as usual:

    >>> network = pyphi.examples.fig4()
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C = subsystem.nodes

Then we'll compute the cause and effect repertoires of mechanism :math:`A` over
purview :math:`ABC`:

    >>> subsystem.cause_repertoire((A,), (A, B, C))
    array([[[ 0.        ,  0.16666667],
            [ 0.16666667,  0.16666667]],
    <BLANKLINE>
           [[ 0.        ,  0.16666667],
            [ 0.16666667,  0.16666667]]])
    >>> subsystem.effect_repertoire((A,), (A, B, C))
    array([[[ 0.0625,  0.0625],
            [ 0.0625,  0.0625]],
    <BLANKLINE>
           [[ 0.1875,  0.1875],
            [ 0.1875,  0.1875]]])

And the unconstrained repertoires over the same (these functions don't take a
mechanism; they only take a purview):

    >>> subsystem.unconstrained_cause_repertoire((A, B, C))
    array([[[ 0.125,  0.125],
            [ 0.125,  0.125]],
    <BLANKLINE>
           [[ 0.125,  0.125],
            [ 0.125,  0.125]]])
    >>> subsystem.unconstrained_effect_repertoire((A, B, C))
    array([[[ 0.09375,  0.09375],
            [ 0.03125,  0.03125]],
    <BLANKLINE>
           [[ 0.28125,  0.28125],
            [ 0.09375,  0.09375]]])

The Earth Mover's distance between them gives the cause and effect information:

    >>> subsystem.cause_info((A,), (A, B, C))
    0.33333191666400036
    >>> subsystem.effect_info((A,), (A, B, C))
    0.24999975000000002

And the minimum of those gives the cause-effect information:

    >>> subsystem.cause_effect_info((A,), (A, B, C))
    0.24999975000000002


Figure 5
~~~~~~~~

**A mechanism generates information only if it has both selective causes and selective effects within the system.**

(A)
```
    >>> network = pyphi.examples.fig5a()
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C = subsystem.nodes

:math:`A` has inputs, so its cause repertoire is selective and it has cause information:

    >>> subsystem.cause_repertoire((A,), (A, B, C))
    array([[[ 0. ,  0. ],
            [ 0. ,  0.5]],
    <BLANKLINE>
           [[ 0. ,  0. ],
            [ 0. ,  0.5]]])
    >>> subsystem.cause_info((A,), (A, B, C))
    0.9999997500000001

But because it has no outputs, its effect repertoire no different from the unconstrained effect repertoire, so it has no effect information:

    >>> np.array_equal(subsystem.effect_repertoire((A,), (A, B, C)),
    ...                subsystem.unconstrained_effect_repertoire((A, B, C)))
    True
    >>> subsystem.effect_info((A,), (A, B, C))
    0.0

And thus its cause effect information is zero.

    >>> subsystem.cause_effect_info((A,), (A, B, C))
    0.0

(B)
```

    >>> network = pyphi.examples.fig5b()
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> A, B, C = subsystem.nodes

Symmetrically, :math:`A` now has outputs, so its effect repertoire is
selective and it has effect information:

    >>> subsystem.effect_repertoire((A,), (A, B, C))
    array([[[ 0.,  0.],
            [ 0.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.],
            [ 0.,  1.]]])
    >>> subsystem.effect_info((A,), (A, B, C))
    0.4999996875

But because it now has no inputs, its cause repertoire is no different from the
unconstrained effect repertoire, so it has no cause information:

    >>> np.array_equal(subsystem.cause_repertoire((A,), (A, B, C)),
    ...                subsystem.unconstrained_cause_repertoire((A, B, C)))
    True
    >>> subsystem.cause_info((A,), (A, B, C))
    0.0

And its cause effect information is again zero.

    >>> subsystem.cause_effect_info((A,), (A, B, C))
    0.0

Figure 6
~~~~~~~~

**Integrated information: The information generated by the whole that is
irreducible to the information generated by its parts.**

    >>> network = pyphi.examples.fig6()
    >>> subsystem = pyphi.Subsystem(range(network.size), network)
    >>> ABC = subsystem.nodes

Here we demonstrate the functions that find the minimum information partition a
mechanism over a purview:

    >>> mip_c = subsystem.mip_past(ABC, ABC)
    >>> mip_e = subsystem.mip_future(ABC, ABC)

These objects contain the :math:`\varphi^{\textrm{MIP}_{\textrm{cause}}}` and
:math:`\varphi^{\textrm{MIP}_{\textrm{effect}}}` values in their respective
``phi`` attributes, and the minimal partitions in their ``partition``
attributes:

    >>> mip_c.phi
    0.499998999999
    >>> mip_c.partition
    (Part(mechanism=(n0,), purview=()), Part(mechanism=(n1, n2), purview=(n0, n1, n2)))
    >>> mip_e.phi
    0.24999975000000002
    >>> mip_e.partition
    (Part(mechanism=(), purview=(n1,)), Part(mechanism=(n0, n1, n2), purview=(n0, n2)))

For more information on these objects, see the API documentation for the
:class:`pyphi.models.Mip` class, or use ``help(mip_c)``. 

Note that the minimal partition found for the past is

.. math::
    frac{A^{c}}{\left[\right]} \times \frac{BC^{c}}{ABC^{p}}
    
rather than the one shown in the figure. However, both partitions result in a
difference of :math:`0.5` between the unpartitioned and partitioned cause
repertoires. So we see that in small networks like this, there can be multiple
choices of partition that yield the same, minimal
:math:`\varphi^{\textrm{MIP}}`. In these cases, which partition the software
chooses is left undefined.
