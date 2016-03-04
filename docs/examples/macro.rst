.. _macro-micro:

Emergence (Macro/Micro)
=======================

* :func:`pyphi.examples.macro_network`

We'll use the :mod:`~pyphi.macro` module to explore alternate spatial scales of
a network. The network under consideration is a 4-node non-deterministic
network, available from the :mod:`~pyphi.examples` module.

    >>> import pyphi
    >>> network = pyphi.examples.macro_network()

The connectivity matrix is all-to-all:

    >>> network.connectivity_matrix
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])

We'll set the state so that nodes are **OFF**.

    >>> state = (0, 0, 0, 0)

At the “micro” spatial scale, we can compute the main complex, and determine
the |big_phi| value:

    >>> main_complex = pyphi.compute.main_complex(network, state)
    >>> main_complex.phi
    0.113889

The question is whether there are other spatial scales which have greater
values of |big_phi|. This is accomplished by considering all possible
coarse-graining of micro-elements to form macro-elements. A coarse-graining of
nodes is any partition of the elements of the micro system. First we'll get a
list of all possible coarse-grainings:

    >>> grains = list(pyphi.macro.all_coarse_grains(network.node_indices))

We start by considering the first coarse grain:

    >>> coarse_grain = grains[0]
    >>> coarse_grain
    CoarseGrain(partition=((0, 1, 2), (3,)), grouping=(((0, 1, 2), (3,)), ((0,), (1,))))

Each ``CoarseGrain`` specifies two fields: the ``partition`` of states into
macro elements, and the ``grouping`` of micro-states into macro-states. Let's
first look at the partition:

    >>> coarse_grain.partition
    ((0, 1, 2), (3,))

There are two macro-elements in this partiion: one consists of
micro-elements ``(0, 1, 2)`` and the other is simply micro-element ``3``.

We must then determine the relationship between micro-elements and
macro-elements. When coarse-graining the system we assume that the
resulting macro-elements do not differentiate the different micro-elements.
Thus any correspondence between states must be stated solely in terms of the
number of micro-elements which are on, and not depend on which micro-elements
are on.

For example, consider the macro-element ``(0, 1, 2)``. We may say that the
macro-element is **ON** if at least one micro-element is on, or if all
micro-elements are on; however, we may not say that the macro-element is **ON**
if micro-element ``1`` is on, because this relationship involves identifying
specific micro-elements.

The ``grouping`` attribute of the ``CoarseGrain`` describes how the state of
micro-elements describes the state of macro-elements:

    >>> grouping = coarse_grain.grouping
    >>> grouping
    (((0, 1, 2), (3,)), ((0,), (1,)))

The grouping consists of two lists, one for each macro-element:

    >>> grouping[0]
    ((0, 1, 2), (3,))

For the first macro-element, this grouping means that the element will be
**OFF** if zero, one or two of its micro-elements are **ON**, and will be
**ON** if all three micro-elements are **ON**.

    >>> grouping[1]
    ((0,), (1,))

For the second macro-element, the grouping means that the element will be
**OFF** if its micro-element is **OFF**, and **ON** if its micro-element is
**ON**.

One we have selected a partition and grouping for analysis, we can create a
mapping between micro-states and macro-states:

    >>> mapping = coarse_grain.make_mapping()
    >>> mapping
    array([0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3])

The interpretation of the mapping uses the **LOLI** convention of indexing (see
:ref:`loli-convention`).

    >>> mapping[7]
    1

This says that micro-state 7 corresponds to macro-state 1:

    >>> pyphi.convert.loli_index2state(7, 4)
    (1, 1, 1, 0)

    >>> pyphi.convert.loli_index2state(1, 2)
    (1, 0)

In micro-state 7, all three elements corresponding to the first macro-element
are **ON**, so that macro-element is **ON**. The micro-element corresponding to
the second macro-element is **OFF**, so that macro-element is **OFF**.

The ``CoarseGrain`` object uses the mapping internally to create a
state-by-state TPM for the macro-system corresponding to the selected partition
and grouping

    >>> coarse_grain.macro_tpm(network.tpm)
    Traceback (most recent call last):
        ...
    pyphi.macro.ConditionallyDependentError

However, this macro-TPM does not satisfy the conditional independence
assumption, so this particular partition and grouping combination is not a valid
coarse-graining of the system. Constructing a |MacroSubsystem| with this
coarse-graining will also raise
:exception:`~pyphi.macro.ConditionallyDependentError`:

Lets consider a different coarse-graining instead.

    >>> coarse_grain = grains[14]
    >>> coarse_grain.partition
    ((0, 1), (2, 3))
    >>> coarse_grain.grouping
    (((0, 1), (2,)), ((0, 1), (2,)))

    >>> mapping = coarse_grain.make_mapping()
    >>> mapping
    array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3])

    >>> coarse_grain.macro_tpm(network.tpm)
    array([[[ 0.09,  0.09],
            [ 1.  ,  0.09]],
    <BLANKLINE>
           [[ 0.09,  1.  ],
            [ 1.  ,  1.  ]]])

We can now construct a |MacroSubsystem| using this coarse-graining:

    >>> macro_subsystem = pyphi.macro.MacroSubsystem(network, state, network.node_indices, coarse_grain=coarse_grain)
    >>> macro_subsystem
    MacroSubsystem((n0, n1))

We can then consider the integrated information of this macro-network and
compare it to the micro-network.

    >>> macro_mip = pyphi.compute.big_mip(macro_subsystem)
    >>> macro_mip.phi
    0.597212

The integrated information of the macro subsystem (:math:`\Phi = 0.597212`) is
greater than the integrated information of the micro system (:math:`\Phi =
0.113889`). We can conclude that a macro-scale is appropriate for this system,
but to determine which one, we must check all possible partitions and all
possible groupings to find the maximum of integrated information across all
scales.

    >>> M = pyphi.macro.emergence(network, state)
    >>> M.emergence
    0.483323
    >>> M.system
    (0, 1, 2, 3)
    >>> M.coarse_grain.partition
    ((0, 1), (2, 3))
    >>> M.coarse_grain.grouping
    (((0, 1), (2,)), ((0, 1), (2,)))

The analysis determines the partition and grouping which results in the maximum
value of integrated information, as well as the emergence (increase in
|big_phi|) from the micro-scale to the macro-scale.
