`combinatorics.sum_of_minimum_among_subsets` computed its subset counts
in int64, silently wrapping once more than 63 values shared an atom —
corrupting `AnalyticalRelations.sum_phi()` (and the campaign scope
report's Σφ_r) on large structures. Counts now stay exact through
int64's range and saturate to `inf` beyond float64's, with zero values
contributing nothing at any size.
`sum_of_minimum_over_size_among_subsets` likewise saturates instead of
raising `OverflowError` past 1023 values.
