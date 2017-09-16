Actual Causation
================

This section demonstrates how to use ``pyphi`` to compute actual causation.

    >>> import pyphi
    >>> from pyphi import actual
    >>> from pyphi.constants import Direction

We will look at how to perform computations over the basic `OR-AND` network
introduced in ``Fig 1`` of the paper.

   >>> network = pyphi.examples.actual_causation()

This is a standard ``pyphi`` |Network|, and we can see its state-by-state TPM.

   >>> pyphi.convert.state_by_node2state_by_state(network.tpm)
   array([[ 1.,  0.,  0.,  0.],
          [ 0.,  1.,  0.,  0.],
          [ 0.,  1.,  0.,  0.],
          [ 0.,  0.,  0.,  1.]])

The ``OR`` gate is element ``0``, and the ``AND`` gate is element ``1`` in the
network.

   >>> OR = 0
   >>> AND = 1

We want to observe both elements at |t-1| and |t|, with ``OR`` on and ``AND``
off in both observations:

   >>> X = Y = (OR, AND)
   >>> X_state = Y_state = (1, 0)

The |Transition| object is the core of all actual causation calculations. To
instantiate a |Transition|, we pass it a |Network|, the state of the network
at |t-1| and |t|, and elements of interest at |t-1| and |t|. Note that
``pyphi`` requires the state to be the state of the entire network,
not just the state of the nodes in the transition.

   >>> transition = actual.Transition(network, X_state, Y_state, X, Y)

Cause and effect repertoires can be obtained for the transition. For example,
as shown on the right side of Figure 2B, we can compute the effect repertoire
to see how |X_t-1 = {OR = 1}| constrains the probability distribution of the
purview |Y_t = {OR, AND}|:

   >>> transition.effect_repertoire((OR,), (OR, AND))
   array([[ 0. ,  0. ],
          [ 0.5,  0.5]])

Similarly, as in Figure 2C, we can compute the cause repertoire of
|Y_t = {OR, AND = 10}| to see how it constrains the purview |X_t-1 = {OR}|:

   >>> transition.cause_repertoire((OR, AND), (OR,))
   array([[ 0.5],
          [ 0.5]])

.. note:: In all |Transition| methods the constraining occurence is passed as
    the ``mechanism`` argument and the constrained occurence is the ``purview``
    argument. This mirrors the terminology introduced in the IIT code.

|Transition| also provides methods for computing cause and effect
ratios. For example, the effect ratio of |X_t-1 = {OR = 1}| constraining
|Y_t = {OR}| (as shown in Figure 3A) is computed as follows:

   >>> transition.effect_ratio((OR,), (OR,))
   0.41503749927884376

The effect ratio of |X_t-1 = {OR = 1}| constraining |Y_t = {AND}| is negative:

   >>> transition.effect_ratio((OR,), (AND,))
   -0.5849625007211563

And the cause ratio of |Y_t = {OR = 1}| constraining |X_t-1 = {OR, AND}|
(Figure 3B) is:

   >>> transition.cause_ratio((OR,), (OR, AND))
   0.41503749927884376

We can evaluate |alpha| for a particular pair of occurences, as in Figure 3C.
For example, to find the irreducible effect ratio of |{OR, AND} -> {OR, AND}|,
we use the ``find_mip`` method:

   >>> link = transition.find_mip(Direction.FUTURE, (OR, AND), (OR, AND))

This returns a |AcMip| object, with a number of useful properties. This
particular MIP is reducible, as we can see by checking the value of |alpha|:

   >>> link.alpha
   0.0

The ``partition`` property shows the minimum information partition that
reduces the occurence and candidate effect:

   >>> link.partition  # doctest: +NORMALIZE_WHITESPACE
    0     1
   ─── ✕ ───
    0     1

Let's look at the MIP for the irreducible occurence |Y_t = {OR, AND}|
constraining |X_t-1 = {OR, AND}| (Figure 3D). This candidate causal link has
positive |alpha|:

   >>> link = transition.find_mip(Direction.PAST, (OR, AND), (OR, AND))
   >>> link.alpha
   0.16992500144231237

   >>> link.partition  # doctest: +NORMALIZE_WHITESPACE
    0     1
   ─── ✕ ───
    0     1

# Note 8: To find the actual cause/effect of a particular occurrence, do this
# (compare Fig. 4, bottom):

   >>> actual_link = transition.find_causal_link(Direction.PAST, (OR, AND))

# Accounts

   >>> account = actual.account(transition)

# Note 9: The irreducibility of the causal account of our transition of
# interest can be evaluated using the following function:

   >>> account = actual.big_acmip(transition)

# Note 10: Find all irreducible accounts within the transition of interest

   >>> all_accounts = pyphi.actual.nexus(network, X_state, Y_state)

# @BO: Probably they are already sorted from largest to smallest, but I'm not sure
   >>> all_accounts = sorted(all_accounts, key=lambda nexus: nexus.alpha, reverse=True)

# Print transition info and Alpha of all irreducible accounts

   >>> transitions_all_accounts = [[n.transition.cause_indices, n.transition.effect_indices, n.alpha] for n in all_accounts]
   >>> print(transitions_all_accounts)
   [[(0,), (0,), 2.0], [(1,), (1,), 2.0], [(0, 1), (0, 1), 0.16992500144231237]]
