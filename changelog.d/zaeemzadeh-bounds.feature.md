Added `pyphi.formalism.iit4.bounds`: the upper bounds on IIT quantities
published in Zaeemzadeh & Tononi (2024, PLOS Comput Biol 20(8):e1012323)
as a standalone research utility. Bound functions return an `UpperBound`
carrying the value and its certificate (`certified`, `assumptions`,
`citation`): Theorem 1 (`distinction_phi_upper_bound`), Lemma 2
(`partition_phi_upper_bound`), the relation bound
(`relation_phi_upper_bound`), the system bound
(`system_phi_upper_bound`), the Bound I/II/III families for the sums of
distinction and relation phi (`sum_phi_distinctions_upper_bound`,
`sum_phi_relations_upper_bound`, `big_phi_upper_bound`, including the
certified Eq 16 growth bound as `bound="GENERAL"`), counting functions
for possible distinctions, relations, and relation faces, and a one-call
`report()`. Functions raise `ValueError` when the active configuration
is outside the confirmed (version, measure) domain. Bound III is
computed in closed form from the supplementary-material formulas and is
validated against the original published experiment code and the real
pipeline on the construction TPM.
