Nonbinary
=========

Here we'll be demonstrating the nonbinary branch of PyPhi by analyzing a real network model, also explored in

    Gomez JD, Mayner WGP, Beheler-Amass M, Tononi G, Albantakis L. Computing
    Integrated Information (Φ) in Discrete Dynamical Systems with
    Multi-Valued Elements. *Entropy*. 2021; 23(1):6.
    https://doi.org/10.3390/e23010006

To begin with, we need to checkout the nonbinary branch from GitHub, found at
https://github.com/wmayner/pyphi/tree/nonbinary, as the implementation is not
currently a part of the primary branch.

.. note::
    As the binary implementation takes advantage of extra assumptions, its
    computations run faster. If you do not need to use the nonbinary version,
    we recommend the binary one instead.

Once on our computer, we can start by importing pyphi:

    >>> import pyphi

Configuration
~~~~~~~~~~~~~

Before computation, there are a few configuration settings we can use to speed up computation:

    >>> pyphi.config.PARTITION_TYPE = 'TRI'
    >>> pyphi.config.MEASURE = 'KLM'
    >>> pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = True
    >>> pyphi.config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = True

Note that 'TRI' as a partition type is an approximation, 'ALL' will give the
true value.

.. important::
    The default EMD measure is not yet implemented for the nonbinary branch,
    so another must be selected. Examples include 'KLM', 'L1', and 'KLD'.

.. warning::
    There may be issues using ``pyphi.config`` to change settings if you are
    not using Linux. If you encounter any, you can instead modify the
    ``pyphi_config.yml`` file in the main directory. Note that this will
    affect everything that uses default settings.

State by State TPM
~~~~~~~~~~~~~~~~~~~~~~~~

In the nonbinary case, the transition probability matrix must be given in
**state-by-state form**, resulting in an S x S matrix, S being the number of
states. For this example, we'll be using the TPM below, based on the TPM in
the paper.

    >>> tpm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

Here, an element in the position |(i, j)| refers to the probability state |i|
at time |t| will transition to state |j| at time |t+1|.

Nonbinary Networks
~~~~~~~~~~~~~~~~~~

An important difference between the standard and nonbinary |Network|
implementation is the num_states_per_node attribute. All that's needed is a
list of integers describing how many states every node in the network has.
For our example, we are working with a network model with three nodes, one
has three states and two have two. For ease of reference, we can call this a
322 network. Then, we can create a |Network| with the system's TPM and number
of states per node.

    >>> base = [3, 2, 2]
    >>> names = ['P', 'C', 'N']
    >>> cm = [[0, 1, 1], [0, 0, 1], [1, 0, 0]]
    >>> network = pyphi.Network(tpm, cm=cm, node_labels=names, num_states_per_node=base)

Optionally, you can include the connectivity matrix with the attribute cm=,
and names with node_labels=. The CM is not necessary for correct results, but
it does improve efficiency.

.. note::
    At the moment, node_labels can only accept lists of **characters**.
    Strings such as 'Mn' are not yet compatible with current methods.

TPM as Pandas Dataframe
~~~~~~~~~~~~~~~~~~~~~~~

Once a TPM is passed to a Network, it can be retrieved in Dataframe format.

    >>> df = network.tpmdf

Now we can work with the tpm as a Dataframe with rows and columns indexed
with a hierarchical MultiIndex. In each index, there is one level per element
and the level values correspond to the element's states. With this, the
pandas groupby() method makes marginalization easy, such as:

    >>> df.groupby('P', axis='columns').sum()
            P	0	1	2
    P	N	C
    0	0	0	0.0	0.0	1.0
    1	0	0	0.0	0.0	1.0
    2	0	0	0.0	0.0	1.0
    0	1	0	0.0	0.0	1.0
    1	1	0	0.0	0.0	1.0
    2	1	0	0.0	0.0	1.0
    0	0	1	1.0	0.0	0.0
    1	0	1	1.0	0.0	0.0
    2	0	1	1.0	0.0	0.0
    0	1	1	1.0	0.0	0.0
    1	1	1	1.0	0.0	0.0
    2	1	1	1.0	0.0	0.0

Computing Phi
~~~~~~~~~~~~~

With the |Network| object generated, standard functions are called the same
way. For example, if we select a state:

    >>> state = (0, 0, 1)

We can create a |Subsystem| by passing both a |Network| and a given state,
and then compute the system irreducibility analysis:

    >>> subsystem = pyphi.Subsystem(network, state)
    >>> sia = pyphi.compute.sia(subsystem)

Finally, we can access the Phi of the given subsystem with the sia's phi
attribute.

    >>> print(sia.phi)
    0.43872200000000006