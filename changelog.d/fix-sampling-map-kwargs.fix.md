Fixed `pyphi.parallel.map_reduce()` crashing with `TypeError` when called with
`map_kwargs` (or multiple item iterables) without an explicit `chunksize`: the
cost-sampling chunksize probe now binds `map_kwargs` and skips sampling for
multi-iterable maps.
