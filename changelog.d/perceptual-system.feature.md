Added the `pyphi.matching` package with `PerceptualSystem` — a system embedded
in an environment via a sensory interface — and `TriggeredTPM`, the system's
fixed-lag response `Pr(S_t | dS_{t-tau} = x)` to each stimulus, constructed by
clamping the interface for `tau_clamp` steps then evolving for the remaining
`tau - tau_clamp`. `PerceptualSystem.triggered_states` gives the
`{stimulus: response_state}` mapping that the cause-effect-structure
computation consumes. `TriggeredTPM.to_pandas()` provides a provisional
labeled view. Binary substrates only.
