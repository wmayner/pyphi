Fixed an axis-ordering bug in `build_triggered_tpm`: the multidimensional
`TriggeredTPM.array` was built with each axis group (sensory, system) in
reversed unit order, so `row()`, `conditional_probability()`,
`marginal_probability()`, and `to_pandas()` attributed probabilities to the
wrong units whenever the sensory interface or the system had two or more
units. `argmax_state` (and therefore triggered states) was unaffected.
Triggering coefficients and perception values computed on multi-unit systems
change accordingly.
