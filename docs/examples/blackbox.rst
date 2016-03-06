
Blackboxing
============

* :func:`pyphi.examples.blackbox_network`

The :mod:`~pyphi.macro` module also provides tools for studying the emergence
of systems using blackboxing.

    >>> import pyphi
    >>> network = pyphi.examples.blackbox_network()

We consider the state where all nodes are off:

    >>> state = (0, 0, 0, 0, 0, 0)
    >>> all_nodes = (0, 1, 2, 3, 4, 5)

The system has minimal |big_phi| without blackboxing:

    >>> subsys = pyphi.Subsystem(network, state, all_nodes)
    >>> pyphi.compute.big_phi(subsys)
    0.215278

 We will consider the blackbox system ``ABC`` and ``DEF``, where ``C`` and
 ``F`` are output elements and ``ABDE`` are hidden within their blackboxes.
 Blackboxing is done with a :class:~`pyphi.macro.Blackbox` object. As with
 :class:~`pyphi.macro.CoarseGrain`, we pass it a partition of micro-elements:

    >>> partition = ((0, 1, 2), (3, 4, 5))
    >>> output_indices = (2, 5)
    >>> blackbox = pyphi.macro.Blackbox(partition, output_indices)

Blackboxes have a few convenience methods. The ``hidden_indices`` property
returns the elements which are hidden within blackboxes:

    >>> blackbox.hidden_indices
    (0, 1, 3, 4)

The ``micro_indices`` property lists all the micro-elements in the box:

    >>> blackbox.micro_indices
    (0, 1, 2, 3, 4, 5)

The ``macro_indices`` property generates a set of indices which index the
blackbox macro-elements. Since there are two blackboxes in our example, and
each has one output element, there are two macro-indices:

    >>> blackbox.macro_indices
    (0, 1)

The ``macro_state`` method converts a state of the micro elements to the state
of the macro-elements. The macro-state of a blackbox system is simply the
state of the system's output elements:

    >>> micro_state = (0, 0, 0, 0, 0, 1)
    >>> blackbox.macro_state(micro_state)
    (0, 1)

Let us also define a time scale over which to perform our analysis:

    >>> time_scale = 2

As in the coarse-graining example, the blackbox and time scale are passed to
|MacroSubsystem|:

    >>> macro_subsystem = pyphi.macro.MacroSubsystem(network, state, all_nodes, blackbox=blackbox, time_scale=time_scale)

We can now compute |big_phi| for this macro system:

    >>> pyphi.compute.big_phi(macro_subsystem)
    0.388889

We find that the macro subsystem has greater integrated information
(:math:`\Phi = 0.388889`) than the micro system (:math:`\Phi =
0.215278`) - the system demonstrates emergence.

WIP...

TODO: demonstrate the blackbox emergence function, once it only does blackboxing

The module provides a utility function for computing all blackboxes of a system:

    >>> blackboxes = pyphi.macro.all_blackboxes(all_nodes)
