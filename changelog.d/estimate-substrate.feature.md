Added `pyphi.estimate`: `estimate_substrate(data, *, regime, prior)` builds
a `SubstratePosterior` (independent Beta posteriors over every TPM cell,
Jeffreys prior by default) from perturbational transition pairs or an
observational trajectory, with a first-class `CoverageReport` recording
which states the data constrained. `SubstratePosterior.sample()` draws an
ordinary `Substrate`, so all existing computations apply to posterior
samples unchanged. The data regime is a required caller assertion.
