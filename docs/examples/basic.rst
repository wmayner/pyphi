Basic Usage
===========

* :func:`pyphi.examples.basic_network`
* :func:`pyphi.examples.basic_subsystem`

Let's make a simple 3-node network and compute its |big_phi|.

To make a network, we need a TPM, current state, past state, and optionally a
connectivity matrix. The TPM can be in more than one form; see the
documentation for :class:`pyphi.network`. Here we'll use the 2-dimensional
state-by-node form.

    >>> import pyphi
    >>> import numpy as np
    >>> tpm = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0],
    ...                 [1, 1, 1], [1, 1, 1], [1, 1, 0]])

The current and past states should be |n|-tuples, where |n| is the number of
nodes in the network, where the |ith| element is the state of the |ith| node in
the network.

    >>> current_state = (1, 0, 0)
    >>> past_state = (1, 1, 0)

The connectivity matrix is a square matrix such that the |i,jth| entry is 1 if
there is a connection from node |i| to node |j|, and 0 otherwise.

    >>> cm = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]])

Now we construct the network itself with the arguments we just created:

    >>> network = pyphi.Network(tpm, current_state, past_state,
    ...                         connectivity_matrix=cm)

The next step is to define a subsystem for which we want to evaluate |big_phi|.
To make a subsystem, we need the indices of subset of nodes which should be
included in it and the network that the subsystem belongs to.

In this case, we want the |big_phi| of the entire network, so we simply include
every node in the network in our subsystem:

    >>> subsystem = pyphi.Subsystem(range(network.size), network)

Now we use :func:`pyphi.compute.big_phi` function to compute the |big_phi| of
our subsystem:

    >>> phi = pyphi.compute.big_phi(subsystem)
    >>> round(phi, pyphi.config.PRECISION)
    2.312498

If we want to take a deeper look at the integrated-information-theoretic
properties of our network, we can access all the intermediate quantities and
structures that are calculated in the course of arriving at a final |big_phi|
value by using :func:`pyphi.compute.big_mip`. This returns a deeply nested
object, |BigMip|, that contains data about the subsystem's constellation of
concepts, cause and effect repertoires, etc.

    >>> mip = pyphi.compute.big_mip(subsystem)

For instance, we can see that this network has 4 concepts:

    >>> len(mip.unpartitioned_constellation)
    4

The documentation for :mod:`pyphi.models` contains description of these
structures.

.. note::

    The network and subsystem discussed here are returned by the
    :func:`pyphi.examples.basic_network` and
    :func:`pyphi.examples.basic_subsystem` functions.

