Added the single-stimulus perception layer to `pyphi.matching`:
`TriggeringCoefficient` and `triggering_coefficient` compute t(x,m) (Eq 5-7)
from a `TriggeredTPM` (which gains `conditional_probability` and
`marginal_probability`), and `Perception(ces, triggered_tpm, stimulus)` exposes
the per-distinction (t*phi_d), per-relation (mean-relata-t * phi_r),
per-fold (t * big_phi_contribution), and total perceptual richness for a
stimulus, as an immutable view that never mutates the structure.
