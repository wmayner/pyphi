Actual Causation
================

This section demonstrates how to use PyPhi to evaluate actual causation, as
described in

    Albantakis L, Marshall W, Hoel E, Tononi G (2019).
    What Caused What? A quantitative Account of Actual Causation Using
    Dynamical Causal Networks.
    *Entropy*, 21 (5), pp. 459.
    `<https://doi.org/10.3390/e21050459>`_

First, we'll import the modules we need:

    >>> import pyphi
    >>> from pyphi import actual, config, Direction

.. only:: never

    This py.test fixture resets PyPhi config back to defaults after running
    this doctest. This will not be shown in the output markup.

    >>> getfixture('restore_config_afterwards')


Configuration
~~~~~~~~~~~~~

Before we begin we need to set some configuration values. The correct way of
partitioning for actual causation is using the ``'ALL'`` partitions setting;
``'TRI'``-partitions are a reasonable approximation. In case of ties the
smaller purview should be chosen. IIT 3.0 style bipartitions will give
incorrect results.

    >>> config.PARTITION_TYPE = 'TRI'
    >>> config.PICK_SMALLEST_PURVIEW = True

When calculating a causal account of the transition between a set of elements
|X| at time |t-1| and a set of elements |Y| at time |t|, with |X| and |Y| being
subsets of the same system, the transition should be valid according to the
system's TPM. However, the state of |X| at |t-1| does not necessarily need to
have a valid previous state so we can disable state validation:

   >>> config.VALIDATE_SUBSYSTEM_STATES = False


Computation
~~~~~~~~~~~

We will look at how to perform computations over the basic `OR-AND` network
introduced in Figure 1 of the paper.

   >>> network = pyphi.examples.actual_causation()

This is a standard PyPhi |Network| so we can look at its TPM:

   >>> pyphi.convert.state_by_node2state_by_state(network.tpm)
   array([[1., 0., 0., 0.],
          [0., 1., 0., 0.],
          [0., 1., 0., 0.],
          [0., 0., 0., 1.]])

The ``OR`` gate is element ``0``, and the ``AND`` gate is element ``1`` in the
network.

   >>> OR = 0
   >>> AND = 1

We want to observe both elements at |t-1| and |t|, with ``OR`` ON and ``AND``
OFF in both observations:

   >>> X = Y = (OR, AND)
   >>> X_state = Y_state = (1, 0)

The |Transition| object is the core of all actual causation calculations. To
instantiate a |Transition|, we pass it a |Network|, the state of the network at
|t-1| and |t|, and elements of interest at |t-1| and |t|. Note that PyPhi
requires the state to be the state of the entire network, not just the state of
the nodes in the transition.

   >>> transition = actual.Transition(network, X_state, Y_state, X, Y)

Cause and effect repertoires can be obtained for the transition. For example,
as shown on the right side of Figure 2B, we can compute the effect repertoire
to see how |X_t-1 = {OR = 1}| constrains the probability distribution of the
purview |Y_t = {OR, AND}|:

   >>> transition.effect_repertoire((OR,), (OR, AND))
   array([[0. , 0. ],
          [0.5, 0.5]])

Similarly, as in Figure 2C, we can compute the cause repertoire of
|Y_t = {OR, AND = 10}| to see how it constrains the purview |X_t-1 = {OR}|:

   >>> transition.cause_repertoire((OR, AND), (OR,))
   array([[0.5],
          [0.5]])

.. note:: In all |Transition| methods the constraining occurence is passed as
    the ``mechanism`` argument and the constrained occurence is the ``purview``
    argument. This mirrors the terminology introduced in the IIT code.

|Transition| also provides methods for computing cause and effect
ratios. For example, the effect ratio of |X_t-1 = {OR = 1}| constraining
|Y_t = {OR}| (as shown in Figure 3A) is computed as follows:

   >>> transition.effect_ratio((OR,), (OR,))
   0.415037

The effect ratio of |X_t-1 = {OR = 1}| constraining |Y_t = {AND}| is negative:

   >>> transition.effect_ratio((OR,), (AND,))
   -0.584963

And the cause ratio of |Y_t = {OR = 1}| constraining |X_t-1 = {OR, AND}|
(Figure 3B) is:

   >>> transition.cause_ratio((OR,), (OR, AND))
   0.415037

We can evaluate |alpha| for a particular pair of occurences, as in Figure 3C.
For example, to find the irreducible effect ratio of |{OR, AND} -> {OR, AND}|,
we use the ``find_mip`` method:

   >>> link = transition.find_mip(Direction.EFFECT, (OR, AND), (OR, AND))

This returns a |AcRepertoireIrreducibilityAnalysis| object, with a number of
useful properties. This particular MIP is reducible, as we can see by checking
the value of |alpha|:

   >>> link.alpha
   0.0

The ``partition`` property shows the minimum information partition that
reduces the occurence and candidate effect:

   >>> link.partition  # doctest: +NORMALIZE_WHITESPACE
    ∅     OR     AND
   ─── ✕ ─── ✕ ───
    ∅     OR     AND

Let's look at the MIP for the irreducible occurence |Y_t = {OR, AND}|
constraining |X_t-1 = {OR, AND}| (Figure 3D). This candidate causal link has
positive |alpha|:

   >>> link = transition.find_mip(Direction.CAUSE, (OR, AND), (OR, AND))
   >>> link.alpha
   0.169925

To find the actual cause or actual effect of a particular occurence, use the
``find_actual_cause`` or ``find_actual_effect`` methods:

   >>> transition.find_actual_cause((OR, AND))
   CausalLink
     α = 0.1699  [OR, AND] ◀━━ [OR, AND]


Accounts
~~~~~~~~

The complete causal account of our transition can be computed with the
``account`` function:

   >>> account = actual.account(transition)
   >>> print(account)  # doctest: +NORMALIZE_WHITESPACE
   <BLANKLINE>
         Account (5 causal links)
   ***********************************
   Irreducible effects
   α = 0.415  [OR] ━━▶ [OR]
   α = 0.415  [AND] ━━▶ [AND]
   Irreducible causes
   α = 0.415  [OR] ◀━━ [OR]
   α = 0.415  [AND] ◀━━ [AND]
   α = 0.1699  [OR, AND] ◀━━ [OR, AND]

We see that this function produces the causal links shown in Figure 4. The
|Account| object is a subclass of ``tuple``, and can manipulated the same:

   >>> len(account)
   5

Irreducible Accounts
~~~~~~~~~~~~~~~~~~~~

The irreducibility of the causal account of our transition of interest can be
evaluated using the following function:

   >>> sia = actual.sia(transition)
   >>> sia.alpha
   0.169925

As shown in Figure 4, the second order occurence |Y_t = {OR, AND = 10}| is
destroyed by the MIP:

   >>> sia.partitioned_account  # doctest: +NORMALIZE_WHITESPACE
   <BLANKLINE>
    Account (4 causal links)
   **************************
   Irreducible effects
   α = 0.415  [OR] ━━▶ [OR]
   α = 0.415  [AND] ━━▶ [AND]
   Irreducible causes
   α = 0.415  [OR] ◀━━ [OR]
   α = 0.415  [AND] ◀━━ [AND]

The partition of the MIP is available in the ``cut`` property:

   >>> sia.cut  # doctest: +NORMALIZE_WHITESPACE
   KCut CAUSE
    ∅     OR    AND
   ─── ✕ ─── ✕ ───
    ∅     OR    AND

To find all irreducible accounts within the transition of interest, use
``nexus``:

   >>> all_accounts = actual.nexus(network, X_state, Y_state)

This computes |big_alpha| for all permutations of of elements in |X_t-1| and
|Y_t| and returns a ``tuple`` of all |AcSystemIrreducibilityAnalysis| objects
with |big_alpha > 0|:

   >>> for n in all_accounts:
   ...     print(n.transition, n.alpha)
   Transition([OR] ━━▶ [OR]) 2.0
   Transition([AND] ━━▶ [AND]) 2.0
   Transition([OR, AND] ━━▶ [OR, AND]) 0.169925

The ``causal_nexus`` function computes the maximally irreducible account for
the transition of interest:

   >>> cn = actual.causal_nexus(network, X_state, Y_state)
   >>> cn.alpha
   2.0
   >>> cn.transition
   Transition([OR] ━━▶ [OR])


Disjunction of conjunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in exploring further, the disjunction of conjunctions
network from Figure 7 is provided as well:

   >>> network = pyphi.examples.disjunction_conjunction_network()
   >>> cn = actual.causal_nexus(network, (1, 0, 1, 0), (0, 0, 0, 1))

The only irreducible transition is from |X_t-1 = C| to |Y_t = D|, with
|big_alpha| of 2.0:

   >>> cn.transition
   Transition([C] ━━▶ [D])
   >>> cn.alpha
   2.0
