Conditional Independence
========================

This example explores the assumption of conditional independence, and the
behaviour of the program when it is not satisfied.

Every state-by-node TPM corresponds to a unique state-by-state TPM which
satisfies the conditional independence assumption. If a state-by-node TPM is
given as input for a network, the program assumes that it is from a system with
the corresponding conditionally independent state-by-state TPM.

When a state-by-state TPM is given as input for a network, the state-by-state
TPM is first converted to a state-by-node TPM. The program then assumes that
the system corresponds to the unique conditionally independent representation
of the state-by-node TPM. **If a non-conditionally independent TPM is given,
the analyzed system will not correspond to the original TPM**. Note that every
deterministic state-by-state TPM will automatically satisfy the conditional
independence assumption.

Consider a system of two binary nodes (|A| and |B|) which do not change if they
have the same value, but flip with probability 50% if they have different
values.

We'll load the state-by-state TPM for such a system from the |examples| module:

   >>> import pyphi
   >>> tpm = pyphi.examples.cond_depend_tpm()
   >>> print(tpm)
   [[ 1.   0.   0.   0. ]
    [ 0.   0.5  0.5  0. ]
    [ 0.   0.5  0.5  0. ]
    [ 0.   0.   0.   1. ]]

This system does not satisfy the conditional independence assumption; given a
past state of ``(1, 0)``, the current state of node |A| depends on whether or
not |B| has flipped.

When creating a network, the program will convert this state-by-state TPM to a
state-by-node form, and issue a warning if it does not satisfy the assumption:

   >>> sbn_tpm = pyphi.convert.state_by_state2state_by_node(tpm)

“The TPM is not conditionally independent. See the conditional independence
example in the documentation for more information on how this is handled.”

   >>> print(sbn_tpm)
   [[[ 0.   0. ]
     [ 0.5  0.5]]
   <BLANKLINE>
    [[ 0.5  0.5]
     [ 1.   1. ]]]

The program will continue with the state-by-node TPM, but since it assumes
conditional independence, the network will not correspond to the original
system.

To see the corresponding conditionally independent TPM, convert the
state-by-node TPM back to state-by-state form:

   >>> sbs_tpm = pyphi.convert.state_by_node2state_by_state(sbn_tpm)
   >>> print(sbs_tpm)
   [[ 1.    0.    0.    0.  ]
    [ 0.25  0.25  0.25  0.25]
    [ 0.25  0.25  0.25  0.25]
    [ 0.    0.    0.    1.  ]]

A system which does not satisfy the conditional independence assumption
exhibits “instantaneous causality.” In such situations, there must be
additional exogenous variable(s) which explain the dependence.

Consider the above example, but with the addition of a third node (|C|) which
is equally likely to be ON or OFF, and such that when nodes |A| and |B| are in
different states, they will flip when |C| is ON, but stay the same when |C| is
OFF.

   >>> tpm2 = pyphi.examples.cond_independ_tpm()
   >>> print(tpm2)
   [[ 0.5  0.   0.   0.   0.5  0.   0.   0. ]
    [ 0.   0.5  0.   0.   0.   0.5  0.   0. ]
    [ 0.   0.   0.5  0.   0.   0.   0.5  0. ]
    [ 0.   0.   0.   0.5  0.   0.   0.   0.5]
    [ 0.5  0.   0.   0.   0.5  0.   0.   0. ]
    [ 0.   0.   0.5  0.   0.   0.   0.5  0. ]
    [ 0.   0.5  0.   0.   0.   0.5  0.   0. ]
    [ 0.   0.   0.   0.5  0.   0.   0.   0.5]]

The resulting state-by-state TPM now satisfies the conditional independence
assumption.

   >>> sbn_tpm2 = pyphi.convert.state_by_state2state_by_node(tpm2)
   >>> print(sbn_tpm2)
   [[[[ 0.   0.   0.5]
      [ 0.   0.   0.5]]
   <BLANKLINE>
     [[ 0.   1.   0.5]
      [ 1.   0.   0.5]]]
   <BLANKLINE>
   <BLANKLINE>
    [[[ 1.   0.   0.5]
      [ 0.   1.   0.5]]
   <BLANKLINE>
     [[ 1.   1.   0.5]
      [ 1.   1.   0.5]]]]

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
