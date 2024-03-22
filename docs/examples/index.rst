Getting started
===============

This page provides a walkthrough of how to use PyPhi in an interactive Python
session. For a theoretical explanation of the computational steps and a complete overview 
of the mathematical formalism please consult the `IIT 4.0 paper <https://doi.org/10.1371/journal.pcbi.1006343.s001>`_.

.. tip::

    A `jupyter notebook
    <https://colab.research.google.com/github/wmayner/pyphi/blob/feature/iit-4.0/docs/examples/IIT_4.0_demo.ipynb>`_  illustrating how to use PyPhi is available as a
    supplement to the `IIT 4.0 paper
    <https://doi.org/10.1371/journal.pcbi.1006343.s001>`_.

To explore the following examples, install `IPython
<https://ipython.org/install.html>`_ by running ``pip install ipython`` on the
command line. Then run it with the command ``ipython``.

Lines of code beginning with ``>>>`` and ``...`` can be pasted directly into
IPython.

----

Basic Usage
===========

Let's apply the formalism of IIT to a simple system of 3 units (also called nodes). This is the same system used in the `IIT 4.0 paper <https://doi.org/10.1371/journal.pcbi.1006343.s001>`_ (Fig. 8C, top).

First we import the required packages and set up PyPhi configuration settings (those specifically needed for this example):

    >>> import pyphi
    >>> import numpy as np
    >>> pyphi.config.PROGRESS_BARS = False
    >>> pyphi.config.PARALLEL = False
    >>> pyphi.config.SHORTCIRCUIT_SIA = False
    >>> pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

Then we have to create our universal substrate; in PyPhi this corresponds to creating a |Network| object. 
To do so, we need a TPM and (optionally) a connectivity matrix. The
TPM can be in more than one form; see the documentation for |Network|. 
Here we'll use the 2-dimensional state-by-node form.

    >>> tpm = np.array([
    ...    [1, 0, 0,],
    ...    [0, 1, 0,],
    ...    [0, 1, 1,],
    ...    [0, 0, 1,],
    ...    [0, 0, 0,],
    ...    [1, 1, 1,],
    ...    [1, 0, 1,],
    ...    [1, 1, 0,]
    ...    ])

The connectivity matrix is a square matrix such that the |i,jth| entry is 1 if
there is a connection from node |i| to node |j|, and 0 otherwise. 
In this case the network is all-to-all connected:

    >>> cm = np.array([
    ...     [1, 1, 1],
    ...     [1, 1, 1],
    ...     [1, 1, 1]
    ... ])

We'll also make labels for the network nodes so that PyPhi's output is easier
to read.

    >>> labels = ('A', 'B', 'C')

Now we construct the network object itself with the arguments we just created:

    >>> network = pyphi.Network(tpm, cm=cm, node_labels=labels)

The next step is to define the candidate complex we want to unfold.
This correponds to creating PyPhi |Subsystem| objects (one for the cause and one for the effect). 
To make a subsystem, we need the network that it belongs to, the state of that
network, and the indices of the subset of nodes which should be included.

The state should be an |n|-tuple, where |n| is the number of nodes in the
network, and where the |ith| element is the state of the |ith| node in the
network.

    >>> state = (1, 0, 0)

If we want the consider the entire universal substrate (network) as our candidate complex (subsystem),
we simply include every node in the network in our subsystem (PyPhi does so automatically if we don't specify any nodes):

    >>> node_indices = (0, 1, 2)
    >>> subsystem_cause = pyphi.Subsystem(network, state, nodes=node_indices, backward_tpm=True)
    >>> subsystem_effect = pyphi.Subsystem(network, state, nodes=node_indices, backward_tpm=False)

Next we compute the φ_s of our candidate complex. We can do so using |backwards.compute.sia()|. 
This returns a nested object, |SystemIrreducibilityAnalysis|, that contains data about the subsystem's
irreducibility, cause and effect repertoires, etc.

    >>> sia = pyphi.backwards.sia(subsystem_cause, subsystem_effect)
    >>> print(sia)
    ┌───────────────────────────────────┐
    │ SystemIrreducibilityAnalysis      │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━        │
    │       Subsystem:  A,B,C           │
    │   Current state:  (1,0,0)         │
    │             φ_s: 2.0              │
    │  Normalized φ_s: 0.4              │
    │           CAUSE:  (0,0,0)         │
    │            II_c: 3.0              │
    │          EFFECT:  (0,1,0)         │
    │            II_e: 3.0              │
    │    #(tied MIPs): 2                │
    │       Partition:                  │
    │                  3 parts: {A,B,C} │
    │                  [[0 0 1]         │
    │                   [1 0 1]         │
    │                   [1 1 0]]        │
    └───────────────────────────────────┘

.. tip::
    Note that if we wanted to apply the postulate of exlusion and find the main complex (the one with maximal φ_s)
    we would have to call the |compute.sia()| function on each possible candidate complex, 
    creating a subsystem for each possible subset of the network.

We can then apply the composition postulate to unfold the cause-effect structure of our (candidate) complex.
A cause-effect structure is composed of distinctions and relations.
First we compute the candidate distinctions:

    >>> candidate_distinctions = pyphi.backwards.compute_combined_ces(subsystem_cause, subsystem_effect)

Then we filter out the distinctions that are not congruent with the cause-effect state of the candidate complex:

    >>> distinctions = candidate_distinctions.resolve_congruence(sia.system_state)

We then compute the relations between those distinctions:

    >>> relations = pyphi.relations.relations(distinctions)

Finally we create and print the cause-effect structure object:

    >>> phi_structure = pyphi.new_big_phi.phi_structure(subsystem=subsystem_effect,distinctions=distinctions,relations=relations,sia=sia)
    >>> print(phi_structure)
    ┌───────────────────────────────────────┐
    │              PhiStructure             │
    │  ════════════════════════════════════ │
    │                Φ: 21.006575494541174  │
    │  #(distinctions):  6                  │
    │            Σ φ_d:  3.1225562489182654 │
    │     #(relations): 60                  │
    │            Σ φ_r: 17.88401924562291   │
    │ ┌───────────────────────────────────┐ │
    │ │ SystemIrreducibilityAnalysis      │ │
    │ │  ━━━━━━━━━━━━━━━━━━━━━━━━━        │ │
    │ │       Subsystem:  A,B,C           │ │
    │ │   Current state:  (1,0,0)         │ │
    │ │             φ_s: 2.0              │ │
    │ │  Normalized φ_s: 0.4              │ │
    │ │           CAUSE:  (0,0,0)         │ │
    │ │            II_c: 3.0              │ │
    │ │          EFFECT:  (0,1,0)         │ │
    │ │            II_e: 3.0              │ │
    │ │    #(tied MIPs): 2                │ │
    │ │       Partition:                  │ │
    │ │                  3 parts: {A,B,C} │ │
    │ │                  [[0 0 1]         │ │
    │ │                   [1 0 1]         │ │
    │ │                   [1 1 0]]        │ │
    │ └───────────────────────────────────┘ │
    └───────────────────────────────────────┘
