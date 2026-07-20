`SweepResult` and `OptimizationResult` now serialize through
`pyphi.serialize`: both gain `save`/`load` (JSON, msgpack, optional gzip),
with their DataFrames embedded as parquet (pyarrow is now a core
dependency). `OptimizationResult.save` previously wrote a summary that
dropped the winning substrate and SIA; it now writes the complete result,
and both types have a load path.
