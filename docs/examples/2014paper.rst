IIT 3.0 Paper (2014)
====================

This section is meant to serve as a companion to the paper `From the
Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory
3.0
<http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588>`_
by Oizumi, Albantakis, and Tononi, and as a demonstration of how to use PyPhi.
Readers are encouraged to follow along and analyze the systems shown in the
figures, in order to become more familiar with both the theory and the
software.

Install `IPython <https://ipython.org/install.html>`_ by running ``pip install
ipython`` on the command line. Then run it with the command ``ipython``.

Lines of code beginning with ``>>>`` and ``...`` can be pasted directly into
IPython.

We begin by importing PyPhi and NumPy:

    >>> import pyphi
    >>> import numpy as np


Figure 1
~~~~~~~~

**Existence: Mechanisms in a state having causal power.**

For the first figure, we'll demonstrate how to set up a network and a candidate
set. In PyPhi, networks are built by specifying a transition probability matrix
and (optionally) a connectivity matrix. (If no connectivity matrix is given,
full connectivity is assumed.) So, to set up the system shown in Figure 1,
we'll start by defining its TPM.

.. note::
    The TPM in the figure is given in **state-by-state** form; there is a row
    and a column for each state. However, in PyPhi, we use a more compact
    representation: **state-by-node** form, in which there is a row for each
    state, but a column for each node. The |i,jth| entry gives the probability
    that the |jth| node is ON in the |ith| state. For more information on how
    TPMs are represented in PyPhi, see :ref:`tpm-conventions`.

In the figure, the TPM is shown only for the candidate set. We'll define the
entire network's TPM. Also, nodes |D|, |E| and |F| are not assigned mechanisms;
for the purposes of this example we will assume they are OR gates. With that
assumption, we get the following TPM (before copying and pasting, see note
below):

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
    This network is already built for you; you can get it from the |examples|
    module with ``network = pyphi.examples.fig1a()``. The TPM can then be
    accessed with ``network.tpm``.

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

Now we can pass the TPM and connectivity matrix as arguments to the network
constructor:

    >>> network = pyphi.Network(tpm, cm=cm)

Now the network shown in the figure is stored in a variable called ``network``.
You can find more information about the network object we just created by
running ``help(network)`` or by consulting the documentation for |Network|.

The next step is to define the candidate set shown in the figure, consisting of
nodes |A|, |B| and |C|. In PyPhi, a candidate set for |big_phi| evaluation is
represented by the |Subsystem| class. Subsystems are built by giving the
network it is a part of, the state of the network, and indices of the nodes to
be included in the subsystem. So, we define our candidate set like so:

    >>> state = (1, 0, 0, 0, 1, 0)
    >>> ABC = pyphi.Subsystem(network, state, [0, 1, 2])

For more information on the subsystem object, see the documentation for
|Subsystem|.

That covers the basic workflow with PyPhi and introduces the two types of
objects we use to represent and analyze networks. First you define the network
of interest with a TPM and connectivity matrix; then you define a candidate set
you want to analyze.


Figure 3
~~~~~~~~

**Information requires selectivity.**

(A)
```

We'll start by setting up the subsytem depicted in the figure and labeling the
nodes. In this case, the subsystem is just the entire network.

    >>> network = pyphi.examples.fig3a()
    >>> state = (1, 0, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C, D = subsystem.node_indices

Since the connections are noisy, we see that |A = 1| is unselective; all
previous states are equally likely:

    >>> subsystem.cause_repertoire((A,), (B, C, D))
    array([[[[0.125, 0.125],
             [0.125, 0.125]],
    <BLANKLINE>
            [[0.125, 0.125],
             [0.125, 0.125]]]])

And this gives us zero cause information:

    >>> subsystem.cause_info((A,), (B, C, D))
    0.0


(B)
```

The same as (A) but without noisy connections:

    >>> network = pyphi.examples.fig3b()
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C, D = subsystem.node_indices

Now, |A|'s cause repertoire is maximally selective.

    >>> cr = subsystem.cause_repertoire((A,), (B, C, D))
    >>> cr
    array([[[[0., 0.],
             [0., 0.]],
    <BLANKLINE>
            [[0., 0.],
             [0., 1.]]]])


Since the cause repertoire is over the purview |BCD|, the first dimension
(which corresponds to |A|'s states) is a singleton. We can squeeze out |A|'s
singleton dimension with

    >>> cr = cr.squeeze()

and now we can see that the probability of |B|, |C|, and |D| having been all ON
is 1:

    >>> cr[(1, 1, 1)]
    1.0

Now the cause information specified by |A = 1| is |1.5|:

    >>> subsystem.cause_info((A,), (B, C, D))
    1.5


(C)
```

The same as (B) but with |A = 0|:

    >>> state = (0, 0, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C, D = subsystem.node_indices

And here the cause repertoire is minimally selective, only ruling out the state
where |B|, |C|, and |D| were all ON:

    >>> subsystem.cause_repertoire((A,), (B, C, D))
    array([[[[0.14285714, 0.14285714],
             [0.14285714, 0.14285714]],
    <BLANKLINE>
            [[0.14285714, 0.14285714],
             [0.14285714, 0.        ]]]])

And so we have less cause information:

    >>> subsystem.cause_info((A,), (B, C, D))
    0.214284


Figure 4
~~~~~~~~

**Information: “Differences that make a difference to a system from its own
intrinsic perspective.”**

First we'll get the network from the |examples| module, set up a subsystem, and
label the nodes, as usual:

    >>> network = pyphi.examples.fig4()
    >>> state = (1, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C = subsystem.node_indices

Then we'll compute the cause and effect repertoires of mechanism |A| over
purview |ABC|:

    >>> subsystem.cause_repertoire((A,), (A, B, C))
    array([[[0.        , 0.16666667],
            [0.16666667, 0.16666667]],
    <BLANKLINE>
           [[0.        , 0.16666667],
            [0.16666667, 0.16666667]]])
    >>> subsystem.effect_repertoire((A,), (A, B, C))
    array([[[0.0625, 0.0625],
            [0.0625, 0.0625]],
    <BLANKLINE>
           [[0.1875, 0.1875],
            [0.1875, 0.1875]]])

And the unconstrained repertoires over the same (these functions don't take a
mechanism; they only take a purview):

    >>> subsystem.unconstrained_cause_repertoire((A, B, C))
    array([[[0.125, 0.125],
            [0.125, 0.125]],
    <BLANKLINE>
           [[0.125, 0.125],
            [0.125, 0.125]]])
    >>> subsystem.unconstrained_effect_repertoire((A, B, C))
    array([[[0.09375, 0.09375],
            [0.03125, 0.03125]],
    <BLANKLINE>
           [[0.28125, 0.28125],
            [0.09375, 0.09375]]])

The Earth Mover's distance between them gives the cause and effect information:

    >>> subsystem.cause_info((A,), (A, B, C))  # doctest: +NUMBER
    0.333332
    >>> subsystem.effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.250000

And the minimum of those gives the cause-effect information:

    >>> subsystem.cause_effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.250000


Figure 5
~~~~~~~~

**A mechanism generates information only if it has both selective causes and
selective effects within the system.**

(A)
```
    >>> network = pyphi.examples.fig5a()
    >>> state = (1, 1, 1)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C = subsystem.node_indices

|A| has inputs, so its cause repertoire is selective and it has cause
information:

    >>> subsystem.cause_repertoire((A,), (A, B, C))
    array([[[0. , 0. ],
            [0. , 0.5]],
    <BLANKLINE>
           [[0. , 0. ],
            [0. , 0.5]]])
    >>> subsystem.cause_info((A,), (A, B, C))  # doctest: +NUMBER
    1.000000

But because it has no outputs, its effect repertoire no different from the
unconstrained effect repertoire, so it has no effect information:

    >>> np.array_equal(subsystem.effect_repertoire((A,), (A, B, C)),
    ...                subsystem.unconstrained_effect_repertoire((A, B, C)))
    True
    >>> subsystem.effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.000000

And thus its cause effect information is zero.

    >>> subsystem.cause_effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.000000

(B)
```

    >>> network = pyphi.examples.fig5b()
    >>> state = (1, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C = subsystem.node_indices

Symmetrically, |A| now has outputs, so its effect repertoire is selective and
it has effect information:

    >>> subsystem.effect_repertoire((A,), (A, B, C))
    array([[[0., 0.],
            [0., 0.]],
    <BLANKLINE>
           [[0., 0.],
            [0., 1.]]])
    >>> subsystem.effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.500000

But because it now has no inputs, its cause repertoire is no different from the
unconstrained effect repertoire, so it has no cause information:

    >>> np.array_equal(subsystem.cause_repertoire((A,), (A, B, C)),
    ...                subsystem.unconstrained_cause_repertoire((A, B, C)))
    True
    >>> subsystem.cause_info((A,), (A, B, C))  # doctest: +NUMBER
    0.000000

And its cause effect information is again zero.

    >>> subsystem.cause_effect_info((A,), (A, B, C))  # doctest: +NUMBER
    0.000000

Figure 6
~~~~~~~~

**Integrated information: The information generated by the whole that is
irreducible to the information generated by its parts.**

    >>> network = pyphi.examples.fig6()
    >>> state = (1, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> ABC = subsystem.node_indices

Here we demonstrate the functions that find the minimum information partition a
mechanism over a purview:

    >>> mip_c = subsystem.cause_mip(ABC, ABC)
    >>> mip_e = subsystem.effect_mip(ABC, ABC)

These objects contain the :math:`\varphi^{\textrm{MIP}}_{\textrm{cause}}` and
:math:`\varphi^{\textrm{MIP}}_{\textrm{effect}}` values in their respective
``phi`` attributes, and the minimal partitions in their ``partition``
attributes:

    >>> mip_c.phi
    0.499999
    >>> mip_c.partition  # doctest: +NORMALIZE_WHITESPACE
     A     B,C
    ─── ✕ ─────
     ∅    A,B,C
    >>> mip_e.phi
    0.25
    >>> mip_e.partition  # doctest: +NORMALIZE_WHITESPACE
     ∅    A,B,C
    ─── ✕ ─────
     B     A,C

For more information on these objects, see the documentation for the
|RepertoireIrreducibilityAnalysis| class, or use ``help(mip_c)``.

Note that the minimal partition found for the cause is

.. math::
    \frac{A^{c}}{\varnothing} \times \frac{BC^{c}}{ABC^{p}},

rather than the one shown in the figure. However, both partitions result in a
difference of |0.5| between the unpartitioned and partitioned cause
repertoires. So we see that in small networks like this, there can be multiple
choices of partition that yield the same, minimal
:math:`\varphi^{\textrm{MIP}}`. In these cases, which partition the software
chooses is left undefined.


Figure 7
~~~~~~~~

**A mechanism generates integrated information only if it has both integrated
causes and integrated effects.**

It is left as an exercise for the reader to use the subsystem methods
``cause_mip`` and ``effect_mip``, introduced in the previous section, to
demonstrate the points made in Figure 7.

To avoid building TPMs and connectivity matrices by hand, you can use the
graphical user interface for PyPhi available online at
http://integratedinformationtheory.org/calculate.html. You can build the
networks shown in the figure there, and then use the **Export** button to
obtain a `JSON <http://en.wikipedia.org/wiki/JSON>`_ file representing the
network. You can then import the file into Python like so:

.. code-block:: python

    network = pyphi.network.from_json('path/to/network.json')


Figure 8
~~~~~~~~

**The maximally integrated cause repertoire over the power set of purviews is
the “core cause” specified by a mechanism.**

    >>> network = pyphi.examples.fig8()
    >>> state = (1, 0, 0)
    >>> subsystem = pyphi.Subsystem(network, state)
    >>> A, B, C = subsystem.node_indices

In PyPhi, the “core cause” is called the *maximally-irreducible cause* (MIC).
To find the MIC of a mechanism over all purviews, use the |Subsystem.mic()|
method:

    >>> mic = subsystem.mic((B, C))
    >>> mic.phi  # doctest: +NUMBER
    0.333334

Similarly, the |Subsystem.mie()| method returns the “core effect” or
*maximally-irreducible effect* (MIE).

For a detailed description of the MIC and MIE objects returned by these
methods, see the documentation for |MIC| or use ``help(subsystem.mic)`` and
``help(subsystem.mie)``.


Figure 9
~~~~~~~~

**A mechanism that specifies a maximally irreducible cause-effect repertoire.**

This figure and the next few use the same network as in Figure 8, so we don't
need to reassign the ``network`` and ``subsystem`` variables.

Together, the MIC and MIE of a mechanism specify a *concept*. In PyPhi, this is
represented by the |Concept| object. Concepts are computed using the
|Subsystem.concept()| method of a subsystem:

    >>> concept_A = subsystem.concept((A,))
    >>> concept_A.phi  # doctest: +NUMBER
    0.166667

As usual, please consult the documentation or use ``help(concept_A)`` for a
detailed description of the |Concept| object.


Figure 10
~~~~~~~~~

**Information: A conceptual structure C (constellation of concepts) is the set
of all concepts generated by a set of elements in a state.**

For functions of entire subsystems rather than mechanisms within them, we use
the |compute| module. In this figure, we see the constellation of concepts of
the powerset of |ABC|'s mechanisms. A constellation of concepts is
represented in PyPhi by a |CauseEffectStructure|. We can compute the
cause-effect structure of the subsystem like so:

    >>> ces = pyphi.compute.ces(subsystem)

And verify that the |small_phi| values match:

    >>> ces.labeled_mechanisms
    (['A'], ['B'], ['C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C'])
    >>> ces.phis  # doctest: +NUMBER
    [0.166667, 0.166667, 0.250000, 0.250000, 0.333334, 0.499999]

The null concept (the small black cross shown in concept-space) is available as
an attribute of the subsystem:

    >>> subsystem.null_concept.phi
    0.0


Figure 11
~~~~~~~~~

**Assessing the conceptual information CI of a conceptual structure
(constellation of concepts).**

Conceptual information can be computed using the function named, as you might
expect, |compute.conceptual_info()|:

    >>> pyphi.compute.conceptual_info(subsystem)  # doctest: +NUMBER
    2.111109


Figure 12
~~~~~~~~~

**Assessing the integrated conceptual information Φ of a constellation C.**

To calculate :math:`\Phi^{\textrm{MIP}}` for a candidate set, we use the
function |compute.sia()|:

    >>> sia = pyphi.compute.sia(subsystem)

The returned value is a large object containing the :math:`\Phi^{\textrm{MIP}}`
value, the minimal cut, the cause-effect structure of the whole set and that of
the partitioned set :math:`C_{\rightarrow}^{\textrm{MIP}}`, the total
calculation time, the calculation time for just the unpartitioned cause-effect
structure, a reference to the subsystem that was analyzed, and a reference to
the subsystem with the minimal unidirectional cut applied. For details see the
documentation for |SystemIrreducibilityAnalysis| or use ``help(sia)``.

We can verify that the :math:`\Phi^{\textrm{MIP}}` value and minimal cut are as
shown in the figure:

    >>> sia.phi
    1.916665
    >>> sia.cut
    Cut [A, B] ━━/ /━━➤ [C]

.. note::

    This ``Cut`` represents removing any connections from the nodes with
    indices ``0`` and ``1`` to the node with index ``2``.

Figure 13
~~~~~~~~~

**A set of elements generates integrated conceptual information Φ only if each
subset has both causes and effects in the rest of the set.**

It is left as an exercise for the reader to demonstrate that of the networks
shown, only **(B)** has |big_phi > 0|.


Figure 14
~~~~~~~~~

**A complex: A local maximum of integrated conceptual information Φ.**

    >>> network = pyphi.examples.fig14()
    >>> state = (1, 0, 0, 0, 1, 0)

To find the subsystem within a network that is the major complex, we use the
function of that name, which returns a |SystemIrreducibilityAnalysis| object:

    >>> major_complex = pyphi.compute.major_complex(network, state)

And we see that the nodes in the complex are indeed |A|, |B|, and |C|:

    >>> major_complex.subsystem.nodes
    (A, B, C)


Figure 15
~~~~~~~~~

**A quale: The maximally irreducible conceptual structure (MICS) generated by a
complex.**

You can use the visual interface at
http://integratedinformationtheory.org/calculate.html to view a conceptual
structure structure in a 3D projection of qualia space. The network in the
figure is already built for you; click the **Load Example** button and select
“IIT 3.0 Paper, Figure 1” (this network is the same as the candidate set in
Figure 1).


Figure 16
~~~~~~~~~

**A system can condense into a major complex and minor complexes that may or
may not interact with it.**

For this figure, we omit nodes :math:`H`, :math:`I`, :math:`J`, :math:`K` and
:math:`L`, since the TPM of the full 12-node network is very large, and the
point can be illustrated without them.

    >>> network = pyphi.examples.fig16()
    >>> state = (1, 0, 0, 1, 1, 1, 0)

To find the maximal set of non-overlapping complexes that a network condenses
into, use |compute.condensed()|:

    >>> condensed = pyphi.compute.condensed(network, state)

We find that there are two complexes: the major complex |ABC| with :math:`\Phi
\approx 1.92`, and a minor complex |FG| with :math:`\Phi \approx 0.069` (note
that there is typo in the figure: |FG|'s |big_phi| value should be |0.069|).
Furthermore, the program has been updated to only consider background
conditions of current states, not previous states; as a result the minor
complex |DE| shown in the paper no longer exists.

    >>> len(condensed)
    2
    >>> ABC, FG = condensed
    >>> (ABC.subsystem.nodes, ABC.phi)
    ((A, B, C), 1.916665)
    >>> (FG.subsystem.nodes, FG.phi)
    ((F, G), 0.069445)

There are several other functions available for working with complexes; see the
documentation for |compute.subsystems()|, |compute.all_complexes()|,
|compute.possible_complexes()|, and |compute.complexes()|.

.. |A = 1| replace:: :math:`A = 1`
.. |A = 0| replace:: :math:`A = 0`
.. |1.5| replace:: :math:`1.5`
.. |0.5| replace:: :math:`0.5`
.. |0.069| replace:: :math:`0.069`
