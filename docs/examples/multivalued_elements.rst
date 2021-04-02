Multi-valued elements
=====================

Here we demonstrate the ``nonbinary`` branch of PyPhi by analyzing the
multivalued p53-Mdm2 network from

    Gomez JD, Mayner WGP, Beheler-Amass M, Tononi G, Albantakis L.
    Computing Integrated Information (Φ) in Discrete Dynamical Systems with Multi-Valued Elements.
    *Entropy*. 2021; 23(1):6.
    https://doi.org/10.3390/e23010006

To begin, we need to install the ``nonbinary`` branch of the PyPhi repository
`on GitHub <https://github.com/wmayner/pyphi/tree/nonbinary>`_:

.. code-block:: bash

    pip install "git+https://github.com/wmayner/pyphi@nonbinary"

.. note::
    As the binary implementation takes advantage of extra assumptions, its
    computations run faster. If you do not need to use the multivalued
    version, we recommend the binary one instead.

Then we can start by importing PyPhi and NumPy:

    >>> import pyphi
    >>> import numpy as np


Configuration
~~~~~~~~~~~~~

Before computation, we need to configure PyPhi with the settings used in the paper:

    >>> pyphi.config.PARTITION_TYPE = 'TRI'
    >>> pyphi.config.MEASURE = 'AID'
    >>> pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = True
    >>> pyphi.config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = True

Note that ``'TRI'`` as a partition type is an approximation; ``'ALL'`` will
consider all possible partitions and give the true value.

.. warning::
    There may be issues using the ``pyphi.config`` to change settings via
    direct assignment if you are not using Linux. If you encounter any, you
    can instead use a ``pyphi_config.yml`` file in the working directory. An
    example with the default settings is available `here
    <https://github.com/wmayner/pyphi/blob/nonbinary/pyphi_config.yml>`_.
    Note that this will affect all computations with PyPhi performed from
    that directory. See :mod:`pyphi.conf` for more information.

.. important::
    The EMD measure is not supported in the ``nonbinary`` branch. IIT's
    measure of choice for mechanism integrated information is the intrinsic
    difference ('AID'), as described in

        Barbosa LS, Marshall W, Albantakis L, Tononi G.
        Mechanism Integrated Information.
        *Entropy*. 2021; 23(3):362.
        https://doi.org/10.3390/e23030362

    Other options include ``'L1'`` and ``'KLD'`` (see
    :attr:`~pyphi.conf.PyphiConfig.MEASURE`).


State-by-state transition probability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the multivalued case, transition probability matrix (TPM) must be given in
**state-by-state form**, resulting in an :math:`S \times S` matrix, where
:math:`S` is the number of states (see :ref:`state-by-state-form`). For this
example, we'll be using the TPM below.

    >>> tpm = np.array([
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ... ])

Here, an element in the position |(i,j)| refers to the probability state |i|
at time |t| will transition to state |j| at time |t+1|.

Multivalued networks
~~~~~~~~~~~~~~~~~~~~

An important difference between the binary and multivalued |Network|
implementations is the ``num_states_per_node`` keyword argument. This is a
list of integers that specify how many states each node in the network has.
For our example, we are working with a network with three nodes: the first
has three states and the others have two. To create a multivalued |Network|,
we provide the TPM and the number of states per node as follows:

    >>> num_states_per_node = [3, 2, 2]
    >>> names = ['P', 'C', 'N']
    >>> cm = [
    ...     [0, 1, 1],
    ...     [0, 0, 1],
    ...     [1, 0, 0],
    ... ]
    >>> network = pyphi.Network(
    ...     tpm,
    ...     cm=cm,
    ...     node_labels=names,
    ...     num_states_per_node=num_states_per_node,
    ... )

Optionally, you can include the connectivity matrix with the keyword argument
``cm``, and labels for the nodes with ``node_labels``. The CM is not
necessary for correct results, but it can greatly improve efficiency if the
network is sparse.

.. note::
    At the moment, ``node_labels`` can only accept iterables of **single
    characters**. Strings such as ``'Mn'`` are not yet supported.


TPM as a Pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the network is created, its TPM can be retrieved as a Pandas DataFrame:

    >>> df = network.tpmdf

In this form, rows and columns are indexed with a hierarchical MultiIndex. In
each index, there is one level per element, with the level values
corresponding to the element's states. The DataFrame's ``groupby()`` method
makes marginalization easy:

    >>> df.groupby('P', axis='columns').sum()  # doctest: +NORMALIZE_WHITESPACE
    P      0  1  2
    P C N
    0 0 0  0  0  1
    1 0 0  0  0  1
    2 0	0  0  0  1
    0 1	0  0  0  1
    1 1	0  0  0  1
    2 1	0  0  0  1
    0 0	1  1  0  0
    1 0	1  1  0  0
    2 0	1  1  0  0
    0 1	1  1  0  0
    1 1	1  1  0  0
    2 1	1  1  0  0

Computing Phi
~~~~~~~~~~~~~

Once the |Network| object is generated, methods are called in the same way as
with the binary implementation. For example, if we select a state,

    >>> state = (0, 0, 1)

we can create a |Subsystem| by passing both the |Network| and the state, and
then compute the system irreducibility analysis:

    >>> subsystem = pyphi.Subsystem(network, state)
    >>> sia = pyphi.compute.sia(subsystem)

Then we can access the |big_phi| of the |Subsystem| with the SIA's phi
attribute.

    >>> sia.phi
    0.43872200000000006
