.. _macro-micro:

Emergence (Macro/Micro)
=======================

* :func:`pyphi.examples.macro_network`

For this example, we will use the ``pprint`` module to display lists in a way
which makes them easer to read.

    >>> from pprint import pprint

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
list of all possible partitions:

    >>> partitions = pyphi.macro.list_all_partitions(network)
    >>> pprint(partitions)
    [[[0, 1, 2], [3]],
     [[0, 1, 3], [2]],
     [[0, 1], [2, 3]],
     [[0, 1], [2], [3]],
     [[0, 2, 3], [1]],
     [[0, 2], [1, 3]],
     [[0, 2], [1], [3]],
     [[0, 3], [1, 2]],
     [[0], [1, 2, 3]],
     [[0], [1, 2], [3]],
     [[0, 3], [1], [2]],
     [[0], [1, 3], [2]],
     [[0], [1], [2, 3]],
     [[0, 1, 2, 3]]]

Lets start by considering the partition ``[[0, 1, 2], [3]]``:

    >>> partition = partitions[0]
    >>> partition
    [[0, 1, 2], [3]]

For this partition there are two macro-elements, one consisting of
micro-elements ``(0, 1, 2)`` and the other is simply micro-element ``3``.

We must then determine the relationship between micro-elements and
macro-elements. An assumption when coarse-graining the system, is that the
resulting macro-elements do not differentiate the different micro-elements.
Thus any correspondence between states must be stated soley in terms of the
number of micro-elements which are on, and not depend on which micro-element
are on.

For example, consider the macro-element ``(0, 1, 2)``. We may say that the
macro-element is **ON** if at least one micro-element is on, or if all
micro-elements are on; however, we may not say that the macro-element is **ON**
if micro-element ``1`` is on, because this relationship involves identifying
specific micro-elements.

To see a list of all possible groupings of micro-states into macro-states:

    >>> groupings = pyphi.macro.list_all_groupings(partition)
    >>> pprint(groupings)
    [[[[0, 1, 2], [3]], [[0], [1]]],
     [[[0, 1, 3], [2]], [[0], [1]]],
     [[[0, 1], [2, 3]], [[0], [1]]],
     [[[0, 2, 3], [1]], [[0], [1]]],
     [[[0, 2], [1, 3]], [[0], [1]]],
     [[[0, 3], [1, 2]], [[0], [1]]],
     [[[0], [1, 2, 3]], [[0], [1]]]]

We will focus on the first grouping in the list.

    >>> grouping = groupings[0]
    >>> grouping
    [[[0, 1, 2], [3]], [[0], [1]]]

The grouping contains two lists, one for each macro-element.

    >>> grouping[0]
    [[0, 1, 2], [3]]

For the first macro-element, this grouping means that the element will be
**OFF** if zero, one or two of its micro-elements are **ON**, and will be
**ON** if all three micro-elements are **ON**.

    >>> grouping[1]
    [[0], [1]]

For the second macro-element, the grouping means that the element will be
**OFF** if its micro-element is **OFF**, and **ON** if its micro-element is
**ON**.

One we have selected a partition and grouping for analysis, we can create a
mapping between micro-states and macro-states:

    >>> mapping = pyphi.macro.make_mapping(partition, grouping)
    >>> mapping
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  2.,
            2.,  2.,  3.])

The interpretation of the mapping uses the **LOLI** convention of indexing (see
:ref:`loli-convention`).

    >>> mapping[7]
    1.0

This says that micro-state 7 corresponds to macro-state 1:

    >>> pyphi.convert.loli_index2state(7, 4)
    (1, 1, 1, 0)

    >>> pyphi.convert.loli_index2state(1, 2)
    (1, 0)

In micro-state 7, all three elements corresponding to the first macro-element
are **ON**, so that macro-element is **ON**. The micro-element corresponding to
the second macro-element is **OFF**, so that macro-element is **OFF**.

Using the mapping, we can then create a state-by-state TPM for the macro-system
corresponding to the selected partition and grouping:

    >>> macro_tpm = pyphi.macro.make_macro_tpm(network.tpm, mapping)
    >>> macro_tpm
    array([[ 0.5838,  0.0162,  0.3802,  0.0198],
           [ 0.    ,  0.    ,  0.91  ,  0.09  ],
           [ 0.5019,  0.0981,  0.3451,  0.0549],
           [ 0.    ,  0.    ,  0.    ,  1.    ]])

This macro-TPM does not satisfy the conditional independence assumption, so
this particular partition and grouping combination is not a valid
coarse-graining of the system:

    >>> pyphi.validate.conditionally_independent(macro_tpm)
    False

In these cases, the object returned :func:`~pyphi.macro.make_macro_network`
function will have a boolean value of ``False``:

    >>> (macro_network, macro_state) = pyphi.macro.make_macro_network(network, state, mapping)
    >>> bool(macro_network)
    False

Lets consider a different partition instead.

    >>> partition = partitions[2]
    >>> partition
    [[0, 1], [2, 3]]

    >>> groupings = pyphi.macro.list_all_groupings(partition)
    >>> grouping = groupings[0]
    >>> grouping
    [[[0, 1], [2]], [[0, 1], [2]]]

    >>> mapping = pyphi.macro.make_mapping(partition, grouping)
    >>> mapping
    array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  2.,
            2.,  2.,  3.])

    >>> (macro_network, macro_state) = pyphi.macro.make_macro_network(network, state, mapping)
    >>> bool(macro_network)
    True

We can then consider the integrated information of this macro-network and
compare it to the micro-network.

    >>> macro_main_complex = pyphi.compute.main_complex(macro_network, macro_state)
    >>> macro_main_complex.phi
    0.86905

The integrated information of the macro system (:math:`\Phi = 0.86905`) is
greater than the integrated information of the micro system (:math:`\Phi =
0.113889`). We can conclude that a macro-scale is appropriate for this system,
but to determine which one, we must check all possible partitions and all
possible groupings to find the maximum of integrated information across all
scales.

    >>> M = pyphi.macro.emergence(network, state)
    >>> M.partition
    [[0, 1], [2, 3]]
    >>> M.grouping
    [[[0, 1], [2]], [[0, 1], [2]]]
    >>> M.emergence
    0.755161

The analysis determines the partition and grouping which results in the maximum
value of integrated information, as well as the emergence (increase in
|big_phi|) from the micro-scale to the macro-scale.
