Added the AC/IIT sign-agreement invariant (B5 follow-on, completing B5):
``TestAcIitSignAgreement`` in ``test/test_cross_formalism_invariants.py`` pins
that actual causation's α and IIT's integration never disagree on sign. Both are
``log2(p/q)`` of the *same* repertoire value at the actual purview state — AC's α
(PMI) reads ``p = transition.probability(...)`` / ``q =
partitioned_probability(...)``, and ``transition.probability`` delegates to the
same ``system.repertoire`` IIT's generalized intrinsic difference uses, where
``gid_at_state = selectivity_at_state * log2(forward/partitioned)`` with the
selectivity repertoire non-negative. The structural identity ``gid =
selectivity * α`` is asserted exactly (to 1e-9) across all
mechanism/purview/partition combinations of a deterministic OR-gate transition
plus a Hypothesis property over random small substrates; ``sign(gid) ==
sign(α)`` follows whenever the selectivity at the state is positive. Guards
against a refactor that diverged the AC probability primitive from the IIT
repertoire.
