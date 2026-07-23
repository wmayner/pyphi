Closed-form subset counts share one overflow policy:
`combinatorics.saturating_pow2` (exact powers through float64's range,
`inf` beyond, never raising). `sum_of_minimum_among_subsets`,
`sum_of_minimum_over_size_among_subsets` (now vectorized),
`sum_of_minimum_of_size_among_subsets`, and the measured relation bounds
all use it; the fixed-degree variant previously raised on counts past
float64 range. Property tests check the two subset sums against exact
big-integer oracles.
