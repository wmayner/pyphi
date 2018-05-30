.. _conditional-independence:

Conditional Independence
========================

Conditional independence is the property of a TPM that *each node's state at
time* |t+1| *must be independent of the state of the others, given the state of
the network at time* |t|:

.. math::
    \Pr(S_{t+1} \mid S_t = s_t) \;=
    \prod_{N \,\in\, S} \Pr(N_{t+1} \mid S_t = s_t)\;,
    \quad \forall \; s_t \in S.

This example explores the assumption of conditional independence, and the
behaviour of the program when it is not satisfied.

Every state-by-node TPM corresponds to a unique state-by-state TPM which
satisfies the conditional independence property (see :ref:`tpm-conventions` for
a discussion of the different TPM forms). If a state-by-node TPM is given as
input for a |Network|, PyPhi assumes that it is from a system with the
corresponding conditionally independent state-by-state TPM.

When a state-by-state TPM is given as input for a |Network|, the state-by-state
TPM is first converted to a state-by-node TPM. PyPhi then assumes that the
system corresponds to the unique conditionally independent representation of
the state-by-node TPM.

.. note::
    Every **deterministic** state-by-state TPM satisfies the conditional
    independence property.

Consider a system of two binary nodes (|A| and |B|) which do not change if they
have the same value, but flip with probability 50% if they have different
values.

We'll load the state-by-state TPM for such a system from the |examples| module:

    >>> import pyphi
    >>> tpm = pyphi.examples.cond_depend_tpm()
    >>> print(tpm)
    [[1.  0.  0.  0. ]
     [0.  0.5 0.5 0. ]
     [0.  0.5 0.5 0. ]
     [0.  0.  0.  1. ]]

This system does not satisfy the conditional independence property; given a
previous state of ``(1, 0)``, the current state of node |A| depends on whether
or not |B| has flipped.

If a conditionally dependent TPM is used to create a |Network|, PyPhi will
raise an error:

    >>> network = pyphi.Network(tpm)
    Traceback (most recent call last):
        ...
    pyphi.exceptions.ConditionallyDependentError: TPM is not conditionally independent.
    See the conditional independence example in the documentation for more info.

To see the conditionally independent TPM that corresponds to the conditionally
dependent TPM, convert it to state-by-node form and then back to state-by-state
form:

    >>> sbn_tpm = pyphi.convert.state_by_state2state_by_node(tpm)
    >>> print(sbn_tpm)
    [[[0.  0. ]
      [0.5 0.5]]
    <BLANKLINE>
     [[0.5 0.5]
      [1.  1. ]]]
    >>> sbs_tpm = pyphi.convert.state_by_node2state_by_state(sbn_tpm)
    >>> print(sbs_tpm)
    [[1.   0.   0.   0.  ]
     [0.25 0.25 0.25 0.25]
     [0.25 0.25 0.25 0.25]
     [0.   0.   0.   1.  ]]

A system which does not satisfy the conditional independence property exhibits
“instantaneous causality.” In such situations, there must be additional
exogenous variable(s) which explain the dependence.

Now consider the above example, but with the addition of a third node (|C|)
which is equally likely to be ON or OFF, and such that when nodes |A| and |B|
are in different states, they will flip when |C| is ON, but stay the same when
|C| is OFF.

    >>> tpm2 = pyphi.examples.cond_independ_tpm()
    >>> print(tpm2)
    [[0.5 0.  0.  0.  0.5 0.  0.  0. ]
     [0.  0.5 0.  0.  0.  0.5 0.  0. ]
     [0.  0.  0.5 0.  0.  0.  0.5 0. ]
     [0.  0.  0.  0.5 0.  0.  0.  0.5]
     [0.5 0.  0.  0.  0.5 0.  0.  0. ]
     [0.  0.  0.5 0.  0.  0.  0.5 0. ]
     [0.  0.5 0.  0.  0.  0.5 0.  0. ]
     [0.  0.  0.  0.5 0.  0.  0.  0.5]]

The resulting state-by-state TPM now satisfies the conditional independence
property.

    >>> sbn_tpm2 = pyphi.convert.state_by_state2state_by_node(tpm2)
    >>> print(sbn_tpm2)
    [[[[0.  0.  0.5]
       [0.  0.  0.5]]
    <BLANKLINE>
      [[0.  1.  0.5]
       [1.  0.  0.5]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[1.  0.  0.5]
       [0.  1.  0.5]]
    <BLANKLINE>
      [[1.  1.  0.5]
       [1.  1.  0.5]]]]

The node indices are ``0`` and ``1`` for |A| and |B|, and ``2`` for |C|:

    >>> AB = [0, 1]
    >>> C = [2]

From here, if we marginalize out the node |C|;

    >>> tpm2_marginalizeC = pyphi.tpm.marginalize_out(C, sbn_tpm2)

And then restrict the purview to only nodes |A| and |B|;

    >>> import numpy as np
    >>> tpm2_purviewAB = np.squeeze(tpm2_marginalizeC[:,:,:,AB])

We get back the original state-by-node TPM from the system with just |A| and
|B|.

    >>> np.all(tpm2_purviewAB == sbn_tpm)
    True
