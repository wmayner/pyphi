`System.proper_cause_marginal` and `System.proper_effect_marginal` now derive
the background from `external_indices` rather than from the complement of
`node_indices`. When the two differ — as in actual-causation
`TransitionSystem`s, where the external set may overlap the system — the
proper marginals previously conditioned on the wrong units, and with
`external_indices=()` (`Transition(..., noise_background=True)`)
`proper_cause_marginal` crashed. External units are conditioned at the
background reference state; substrate units neither in the system nor
external are marginalized uniformly. For plain `System`s, where the external
set is the complement of the system, results are unchanged.
