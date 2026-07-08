`SubstratePosterior`, `CoverageReport`, and `PhiPosterior` round-trip
through `pyphi.serialize` (JSON and msgpack), including the raw posterior
parameters and per-draw Φ samples, so estimation results can be stored and
re-analyzed without recomputation.
