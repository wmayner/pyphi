Added `pyphi.estimate.phi_posterior`: Monte Carlo propagation of a
`SubstratePosterior` through the SIA, returning a `PhiPosterior` that
reports the mixture honestly — `p_positive` (the probability the system is
integrated at all), unconditional and conditional quantiles, the raw Φ
samples, and the complex-identity categorical (which unit set is maximal,
per sample). A `PhiPosterior` cannot be coerced to a bare float; the error
names the honest summaries and, when state coverage is partial, the
unconstrained states.
