`Substrate.sia()` results are now deterministic across runs. Previously,
when multiple partitions tied at the MIP minimisation key, the
first-encountered tied partition under `MapReduce` iteration won — a
race that surfaced on symmetric substrates (fully-connected lattices,
the `big_subsys_all_complete` fixture). A structural tie-break on the
induced edge cut now selects the canonical partition. The new
`sia_tie_resolution` config option exposes the ordering;
the default is `["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]`.
