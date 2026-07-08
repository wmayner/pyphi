The state-by-state path of `pyphi.dynamics.simulate` now samples each step by
inverse-CDF over a precomputed cumulative distribution (`numpy.searchsorted`)
instead of a per-step `pandas.Series.sample`, roughly 120x faster per step
(~1 us vs ~120 us) while preserving full-joint sampling.
