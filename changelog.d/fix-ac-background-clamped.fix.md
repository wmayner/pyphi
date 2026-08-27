Actual-causation background conditions now match the paper's causal model in
both directions: units outside the cause set have their inputs to the
transition fixed at the observed before-state (Albantakis et al. 2019,
Section 3.3 — the background U is set to u throughout), for cause repertoires
as well as effect repertoires. Previously the cause direction integrated the
background's past states under the posterior implied by the observed present,
which deviates whenever the background's own dynamics are informative
(Figure 8B's cause link came out 1.2345 bits instead of 3.0 on such
backgrounds; every published example has static backgrounds, where the two
readings coincide). `noise_background=True` now marginalizes background
inputs uniformly on the cause side too, as documented. `System` gains a
`background_state` field for conditioning external units at a state other
than the evaluation state.
