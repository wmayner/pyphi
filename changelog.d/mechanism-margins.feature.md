Added mechanism-level selection margins, completing margin reporting across
every selection in the IIT 4.0 pipeline:
`RepertoireIrreducibilityAnalysis.partition_margin` (normalized-φ gap between
the mechanism MIP and the best competing partition; `None` when there is no
competitor or the sweep short-circuited), `purview_margin` on the MICE (φ gap
to the best competing purview — a measure of how decisively the structural
choice was made, since purview selection is value-continuous), the
already-computed specified-state margins surfaced as `state_margin`, and
`effectively_tied` on RIA, MICE, and `Distinction`. Margins appear in
`explain()` findings, the FULL-verbosity cards, `to_pandas()`, and round-trip
through serialization and relabeling (older payloads load with margins
absent).
