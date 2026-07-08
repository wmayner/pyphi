Added selection-margin reporting to the IIT 4.0 SIA: `partition_margin`
(normalized-φ gap between the MIP and the best competing system partition),
per-direction specified-state margins
(`StateSpecification.state_margin`, from the retained second-best state's
intrinsic information), and `SystemIrreducibilityAnalysis.effectively_tied`
(whether any selection margin is within `config.numerics.precision` of
zero). Margins are computed from values the SIA search already produces,
surface in `explain()` findings, the `FULL`-verbosity card, and
`to_pandas()`, and round-trip through serialization (older serialized
results load with margins absent).
